import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

from .config import *

# This file contains code for addressing questions in the data


def gdf_to_df(gdf):
    """
    Takes a geopandas dataframe and returns a pandas dataframe
    that is more suitable for ML.
    Args:
        gdf: geopandas dataframe.
    Returns:
        df: Pandas dataframe.
    """
    df = pd.DataFrame(gdf)

    df['latitude'] = df.apply(lambda row: row.geometry.centroid.y, axis=1)
    df['longitude'] = df.apply(lambda row: row.geometry.centroid.x, axis=1)

    return df


def count_pois_near_coordinates(pois, tags: dict) -> dict:
    """
    Count each type of Points of Interest (POI).
    Args:
        pois (gdf): Points of interest.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    df = gdf_to_df(pois)

    poi_counts = {}

    for tag, value in tags.items():
        if value == True:
            if tag not in df.columns:
                poi_counts[tag] = 0
            poi_counts[tag] = df[tag].notnull().sum()
        else:
            sub_tags = value if isinstance(value, list) else [value]
            for sub_tag in sub_tags:
                poi_counts[f"{tag}:{sub_tag}"] = (df[tag] == sub_tag).sum()

    return poi_counts


def plot_distance_matrix_heatmap(names, data, label):
    """
    Plot a distance matrix for ehd provided data.
    Args:
        names (list): List of strings to label the rows and columns.
        data (df): Dataframe where rows are vectors.
        label (string): Label for x and y axes.
    """

    dist_matrix = euclidean_distances(data)
    pd.DataFrame(dist_matrix, columns=names, index=names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, annot=True, cmap="viridis",
                cbar=True, xticklabels=names, yticknames=names)
    plt.xlabel(label)
    plt.ylabel(label)
    plt.show()


def join_osm_data_to_pp_data(osm_data, pp_data):
    """Join OSM data and PP data on housenumber, street, and postcode"""

    # Create filtered OSM df with min, max house number columns
    # and the street column capitalised to match pp_data

    with_house_no = osm_data["addr:housenumber"].str.isnumeric()
    # with_house_range = osm_data["addr:housenumber"].str.match(r"^[\d]+-[\d]+")

    osm_matchable = osm_data[with_house_no].copy()

    house_no = osm_matchable["addr:housenumber"]
    osm_matchable["min"] = house_no
    osm_matchable["max"] = house_no

    # split_min_max = house_no[with_house_range].str.split("-", expand=True)
    # osm_matchable.loc[with_house_range, ["min", "max"]] = split_min_max.values

    osm_matchable["min"] = pd.to_numeric(osm_matchable["min"])
    osm_matchable["max"] = pd.to_numeric(osm_matchable["max"])

    osm_matchable["street"] = osm_matchable["addr:street"].str.upper()
    osm_matchable["postcode"] = osm_matchable["addr:postcode"]

    # Create filtered PP df with numeric house number

    house_number = pp_data["primary_addressable_object_name"].str.isnumeric()
    pp_matchable = pp_data[house_number &
                           pp_data["street"].notna() & pp_data["postcode"].notna()]

    pp_matchable["housenumber"] = pd.to_numeric(
        pp_matchable["primary_addressable_object_name"])

    # Join the two dataframes

    merged = pd.merge(osm_matchable, pp_matchable, on=["postcode", "street"])
    merged = merged.loc[(merged["housenumber"] >= merged["min"]) & (
        merged["housenumber"] <= merged["max"])]

    return merged
