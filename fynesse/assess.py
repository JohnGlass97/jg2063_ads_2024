import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

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
        if tag not in df.columns:
            poi_counts[tag] = 0
        elif value == True:
            poi_counts[tag] = df[tag].notnull().sum()
        else:
            for sub_tag in value:
                 poi_counts[f"{tag}:{sub_tag}"] = (df[tag]==sub_tag).sum()

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
    sns.heatmap(dist_matrix, annot=True, cmap="viridis", cbar=True, xticklabels=names, yticknames=names)
    plt.xlabel(label)
    plt.ylabel(label)
    plt.show()