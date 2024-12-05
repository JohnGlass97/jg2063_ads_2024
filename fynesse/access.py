import csv
import io
import zipfile
import pymysql
import requests
import osmnx as ox
import multiprocessing as mp
import pandas as pd
import osmium

from .config import *

# This file accesses the data


def download_price_paid_data(year_from: int, year_to: int) -> None:
    """Download UK house price data for given year range inclusive as CSV files."""

    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"

    for year in range(year_from, (year_to+1)):
        print(f"Downloading data for year: {year}")
        for part in range(1, 3):
            file_name = f"/pp-{year}-part{part}"
            response = requests.get(base_url + file_name)
            if response.status_code == 200:
                local_file_name = f"./{file_name}.csv"
                with open(local_file_name, "wb") as file:
                    file.write(response.content)


def create_connection(user: str, password: str, host: str, database: str, port=3306) -> pymysql.Connection | None:
    """ Create a database connection to the MariaDB database
        specified by the host url and database name."""

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


def get_df_from_query(conn: pymysql.Connection, query: str) -> pd.DataFrame:
    """Run a query on the database and return the result as a pandas dataframe."""

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=[x[0].split(".")[-1]
                                         for x in cur.description])
    return df


def housing_upload_join_data(conn: pymysql.Connection, year: int) -> None:
    """Upload the price paid data for the given year to the database and join it with the postcode data."""

    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    with conn.cursor() as cur:
        print("Selecting data for year: " + str(year))
        cur.execute("""SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type,
                    pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude
                    FROM (
                        SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city,
                        district, county FROM pp_data """ +
                    f"WHERE date_of_transfer BETWEEN {start_date} AND {end_date}" +
                    ") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode")
        rows = cur.fetchall()

        csv_file_path = "output_file.csv"

        # Write the rows to the CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the data rows
            csv_writer.writerows(rows)
        print("Storing data for year: " + str(year))
        cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path +
                    "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
        conn.commit()

    print("Data stored for year: " + str(year))


def upload_df_to_db(conn: pymysql.Connection, df: pd.DataFrame, table_name: str) -> None:
    """Upload a pandas dataframe to the database as a table."""

    csv_file_path = "output_file.csv"
    df.to_csv(csv_file_path, header=False)

    with conn.cursor() as cur:
        cur.execute(f"LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE `{table_name}` " +
                    "FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY " +
                    "'' TERMINATED BY '\n';")
        conn.commit()


def tag_dict_to_tag_set(tags: dict) -> set:
    """Convert a dictionary of tags to a set of tags."""

    tag_set = set()
    for tag, value in tags.items():
        if value == True:
            tag_set.add(tag)
        else:
            sub_tags = value if isinstance(value, list) else [value]
            for sub_tag in sub_tags:
                tag_set.add(f"{tag}:{sub_tag}")

    return tag_set


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
    radius = round(distance_km * 1000)

    features = ox.features_from_point(point, tags, radius)
    print("-", end="")  # To signify progress

    return features


def parallel_fetch_pois(locations_dict: dict, tags: dict, distance_km: float = 1.0, thread_count: int = 5) -> dict:
    """
    Fetch Points of Interest (POIs) near each given pair of coordinates within a specified distance.
    Args:
        locations_dict (dict): A dictionary of names to latitude longitude tuples.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        gdf_dict: A dict of geopandas dataframe of the POIs.
    """
    keys, values = zip(*locations_dict.items())
    n = len(locations_dict) if thread_count < 1 else thread_count

    inputs = [(x, y, tags, distance_km) for x, y in values]

    with mp.Pool(n) as p:
        results = p.starmap(fetch_pois, inputs)

    return {loc: res for loc, res in zip(keys, results)}


def parallel_map_dict(input_dict: dict, f, thread_count: int) -> dict:
    """Map a function f over a dictionary in parallel using thread_count threads."""
    keys, values = zip(*input_dict.items())

    with mp.Pool(thread_count) as p:
        results = p.starmap(f, values)

    return {k: v for k, v in zip(keys, results)}


def bbox_of(lat: float, long: float, distance_km: float) -> tuple:
    """Returns an approx bbox of the form (N, S, E, W) with specified width and height.
    It is only accurate at the equator"""

    d = distance_km / 111 / 2

    return (lat+d, lat-d, long+d, long-d)


def get_price_paid_data_in_regions(conn: pymysql.Connection, bbox: tuple, from_date, columns) -> pd.DataFrame:
    """Use provided DB connection to get price paid data within the given bounding box.
    Note that the bbox is approximated and only accurate at the equator."""

    subquery = (f"SELECT {', '.join(columns)}, pp.postcode AS postcode, " + """
        ROW_NUMBER() OVER (PARTITION BY pp.postcode, street, primary_addressable_object_name ORDER BY date_of_transfer DESC) AS row_num
        FROM pp_data pp JOIN postcode_data po ON pp.postcode = po.postcode
        """ + f"WHERE po.latitude BETWEEN {bbox[1]} AND {bbox[0]} AND po.longitude BETWEEN {bbox[3]} AND {bbox[2]} " +
                f"AND date_of_transfer >= '{from_date}'")

    query = f"SELECT {', '.join(columns)}, postcode FROM ({subquery})" + \
        "AS ranked WHERE row_num = 1;"

    return get_df_from_query(conn, query)


def get_buildings_with_addresses(bbox: tuple):
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


def download_zip_data(url: str, extract_dir) -> None:
    """Download a zip file from the given URL and extract it to the given directory."""

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Files already exist at: {extract_dir}.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Files extracted to: {extract_dir}")


def fetch_postcode_mappings(levels: list[str]) -> pd.DataFrame:
    """Fetch the mappings from postcodes to the given levels."""

    download_zip_data(
        "https://www.arcgis.com/sharing/rest/content/items/ea2f70bb71a54d818232e987cc1d28b9/data",
        "postcode_mappings")

    file_path = "postcode_mappings/pcd_oa_lsoa_msoa_ltla_utla_rgn_ctry_ew_may_2021_lu_v2.csv"
    postcode_mappings_df = pd.read_csv(file_path)

    all_cols = list(postcode_mappings_df.columns)
    cols = ["pcd"] + \
        [c for c in all_cols if any([c.startswith(l) for l in levels])]
    postcode_mappings_df = postcode_mappings_df[cols].set_index("pcd")

    return postcode_mappings_df


def download_census_data(code: str, base_dir="") -> None:
    """Download the 2021 census data for the given code and extract it to the base directory."""

    url = "https://www.nomisweb.co.uk/output/census/2021/census2021-" + \
        f"{code.lower()}.zip"
    extract_dir = os.path.join(
        base_dir, os.path.splitext(os.path.basename(url))[0])

    download_zip_data(url, extract_dir)


def load_census_data(code: str, level: str) -> pd.DataFrame:
    """Load the 2021 census data for the given code and level."""
    return pd.read_csv(f"census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv")


def fetch_age_distributions(level: str) -> pd.DataFrame:
    """Fetch the age distribution data from the 2021 census."""

    download_census_data("TS007")  # Age by single year of age

    age_df = load_census_data("TS007", level)
    age_df = age_df.drop(age_df.columns[[
                         0, 3, 4, 10, 16, 23, 28, 34, 45, 61, 77, 88, 99, 115]], axis=1).set_index("geography")
    age_df.columns = list(range(100))
    return age_df


def fetch_age_bands(level: str) -> pd.DataFrame:
    """Fetch the age band data from the 2021 census."""

    download_census_data("TS007")  # Age by bands

    age_df = load_census_data("TS007", level)
    age_df = age_df.iloc[:, [1, 2, 3, 4, 10, 16, 23, 28, 34,
                             45, 61, 77, 88, 99, 115]].set_index("geography")
    age_df.columns = [x.replace("Age: ", "") for x in age_df.columns]
    return age_df


def fetch_ns_sec(level: str) -> pd.DataFrame:
    """Fetch the National Statistics Socio-economic Classification (NS-SEC) data from the 2021 census."""

    download_census_data("TS062")

    ns_sec_df = load_census_data("TS062", level)
    ns_sec_df = ns_sec_df.drop("date", axis=1).set_index("geography")
    ns_sec_df.columns = [x.replace(
        "National Statistics Socio-economic Classification (NS-SEC): ", "") for x in ns_sec_df.columns]
    return ns_sec_df


def fetch_general_health(level: str) -> pd.DataFrame:
    """Fetch the general health data from the 2021 census."""

    download_census_data('TS037')

    general_health_df = load_census_data('TS037', level)
    general_health_df = general_health_df.drop(
        "date", axis=1).set_index("geography")
    general_health_df.columns = [
        x.replace("General health: ", "") for x in general_health_df.columns]
    return general_health_df


def fetch_output_area_data() -> pd.DataFrame:
    """Fetch Census Output Area data including coordinates."""

    url = "https://open-geography-portalx-ons.hub.arcgis.com/" + \
        "api/download/v1/items/6beafcfd9b9c4c9993a06b6b199d7e6d/csv?layers=0"

    response = requests.get(url)
    response.raise_for_status()

    oa_data_df = pd.read_csv(io.StringIO(response.text))
    oa_data_df = oa_data_df.drop(
        "FID", axis=1).set_index("OA21CD")

    return oa_data_df


class PbfTagFilterHandler(osmium.SimpleHandler):
    def __init__(self, tag_set: set[str]):
        super().__init__()
        self.osm_features = []
        self._tag_set = tag_set

    def node(self, x):
        self._process_element(x)

    def way(self, x):
        self._process_element(x)

    def relation(self, x):
        self._process_element(x)

    def _process_element(self, x):
        if not (hasattr(x, "location") and x.location.valid):
            return
        for k, v in dict(x.tags).items():
            for include_sub_tag in [True, False]:
                tag = f"{k}:{v}" if include_sub_tag else k

                if tag in self._tag_set:
                    self.osm_features.append({
                        "id": x.id,
                        "latitude": x.location.lat,
                        "longitude": x.location.lon,
                        "tag": tag,
                    })


def pbf_to_dataframe(pbf_file: str, filter_tag_set: set[str]) -> pd.DataFrame:
    """Convert a PBF file to a pandas dataframe filtering with the given tags."""

    handler = PbfTagFilterHandler(filter_tag_set)
    handler.apply_file(pbf_file)

    df = pd.DataFrame(handler.osm_features).set_index("id")
    return df
