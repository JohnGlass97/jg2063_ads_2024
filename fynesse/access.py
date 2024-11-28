import csv
import pymysql
import requests
import osmnx as ox
import multiprocessing as mp
import pandas as pd

from .config import *

# This file accesses the data


def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print(f"Downloading data for year: {year}")
        for part in range(1, 3):
            url = base_url + \
                file_name.replace("<year>", str(year)).replace(
                    "<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)


def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def housing_upload_join_data(conn, year):
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    cur = conn.cursor()
    print('Selecting data for year: ' + str(year))
    cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' +
                start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
    rows = cur.fetchall()

    csv_file_path = 'output_file.csv'

    # Write the rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the data rows
        csv_writer.writerows(rows)
    print('Storing data for year: ' + str(year))
    cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path +
                "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    conn.commit()
    print('Data stored for year: ' + str(year))


def fetch_pois(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0):
    """
    Fetch Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        gdf: A geopandas dataframe of the POIs.
    """
    point = latitude, longitude
    radius = distance_km * 1000

    features = ox.features_from_point(point, tags, radius)
    print("-", end="")  # To signify progress

    return features


def parallel_fetch_pois(locations_dict: dict, tags: dict, distance_km: float = 1.0):
    """
    Fetch Points of Interest (POIs) near each given pair of coordinates within a specified distance.
    Args:
        locations_dict (dict): A dictionary of names to latitude longitude tuples.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        gdf_dict: A dict of geopandas dataframe of the POIs.
    """
    with mp.Pool(len(locations_dict)) as p:
        inputs = [(x, y, tags, distance_km)
                  for (x, y) in locations_dict.values()]

        results = p.starmap(fetch_pois, inputs)
    return [a for a in zip(locations_dict.keys(), results)]


def bbox_of(lat, long, distance_km):
    """Returns an approx bbox of the form (N, S, E, W) with specified width and height."""

    d = distance_km / 111 / 2
    return (lat+d, lat-d, long+d, long-d)


def get_price_paid_data_in_regions(conn, bbox, from_date, columns):
    """Use provided DB connection to get price paid data within the given bounding box."""

    cur = conn.cursor()
    subquery = f"SELECT {', '.join(columns)}, pp.postcode AS postcode, " + \
        "ROW_NUMBER() OVER (PARTITION BY pp.postcode, street, primary_addressable_object_name ORDER BY date_of_transfer DESC) AS row_num " + \
        "FROM pp_data pp JOIN postcode_data po ON pp.postcode = po.postcode " + \
        f"WHERE po.latitude BETWEEN {bbox[1]} AND {bbox[0]} AND po.longitude BETWEEN {bbox[3]} AND {bbox[2]} " + \
        f"AND date_of_transfer >= '{from_date}'"

    cur.execute(f"SELECT {', '.join(columns)}, postcode FROM ({
                subquery}) AS ranked WHERE row_num = 1;")

    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=[x[0].split(".")[-1]
                      for x in cur.description])
    return df


def get_buildings_with_addresses(bbox):
    """Returns OSM buildings with housenumber, street, and postcode."""

    tags = {
        "building": True,
    }

    buildings = ox.features_from_bbox(bbox=bbox, tags=tags)
    buildings["area"] = buildings.to_crs(
        {'init': 'epsg:32633'})["geometry"].area

    COLUMNS = ["geometry", "addr:city", "addr:housenumber",
               "addr:postcode", "addr:street", "name", "addr:housename",
               "addr:country", "nodes", "building", "ways", "type", "area",
               ]

    buildings = buildings[COLUMNS]

    has_full_address = (buildings["addr:housenumber"].notna()) & \
        (buildings["addr:street"].notna()) & \
        (buildings["addr:postcode"].notna())

    with_address = buildings[has_full_address]

    return with_address
