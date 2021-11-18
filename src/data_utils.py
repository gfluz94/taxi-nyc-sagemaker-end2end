from typing import List, Tuple
import numpy as np
import pandas as pd


def haversine(pickup: Tuple[float], dropoff: Tuple[float], R: float = 6371) -> float:
    """
    Method which calculates the distance between (lat, lon) pairs on Earth.
    
    Args: 
        pickup (tuple of float): pickup (lat, lon)
        dropoff (tuple of float): dropoff (lat, lon)
    
    Returns: 
        A float number representing the distance, in km.
    """
    phi1, lambda1 = [np.radians(v) for v in pickup]
    phi2, lambda2 = [np.radians(v) for v in dropoff]
    
    delta_lambda = lambda2 - lambda1
    delta_phi = phi2 - phi1
        
    d = 2*R*np.arcsin( np.sqrt(np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2) )

    return d


def calculate_distance(pickup_lat: str, pickup_long: str,
                       dropoff_lat: str, dropoff_long: str) -> np.array:
    """
    Method encapsulates a function to be applied in a pandas DF.
    
    Args: 
        pickup_lat (str): pickup latitude column name
        pickup_long (str): pickup longitude column name
        dropoff_lat (str): dropoff latitude column name
        dropoff_long (str): dropoff longitude column name
    
    Returns: 
        A wrapped function that applies Haversine Distance to a Pandas dataframe.
    """
    def wrapper_calculate_distance(df: pd.DataFrame):
        pickup = (df[pickup_lat], df[pickup_long])
        dropoff = (df[dropoff_lat], df[dropoff_long])
        return haversine(pickup, dropoff)
    
    return wrapper_calculate_distance


def engineer_features(data: pd.DataFrame, feature_cross: bool = True) -> pd.DataFrame:
    """
    Function that performs feature engineering to enrich a dataset.
    Transformations include haversine distance between (lat, lon) pairs and datetime information extraction.
    
    Args: 
        data (pandas dataframe): original dataset
        feature_cross (bool): True, if feature cross should be applied to day of week and 
        hour of day.
    
    Returns: 
        Dataset including new features
    """
    data.loc[:, "distance_ride"] = data.apply(
        calculate_distance("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"),
        axis=1
    )
    data.loc[:, "pickup_datetime"] = pd.to_datetime(data.loc[:, "pickup_datetime"])
    data.loc[:, "pickup_datetime_nyc"] = data.loc[:, "pickup_datetime"] - pd.Timedelta(hours=4)
    data.loc[:, "hour_ride"] = data.loc[:, "pickup_datetime_nyc"].dt.hour.astype("category")
    data.loc[:, "period_ride"] = (data.loc[:, "hour_ride"]
                                      .apply(lambda x: "AM" if x < 12 else "PM")
                                      .astype("category")
                                 )
    data.loc[:, "day_ride"] = data.loc[:, "pickup_datetime_nyc"].dt.strftime("%A").astype("category")
    if feature_cross:
        data.loc[:, "hour_day_ride"] = (
            data.loc[:, "day_ride"].astype(str) + "_" + data.loc[:, "hour_ride"].astype(str)
        )
        data = data.drop(columns=["day_ride", "hour_ride"])
        data.loc[:, "hour_day_ride"] = data.loc[:, "hour_day_ride"]
    return data


def prepare_data(data: pd.DataFrame, feature_engineering: bool = True, 
                 drop_columns: List[str] = None) -> pd.DataFrame:
    """
    Function that prepares data for the downstream prediction pipeline.
    Feature engineering and column removal can both be performed.
    
    Args: 
        data (pandas dataframe): original dataset
        feature_engineering (bool): True, if feature engineering shall be performed
        drop_columns (list of str): columns that are to be dropped
    
    Returns: 
        Dataset including new features
    """
    output = data.copy()
    if feature_engineering:
        output = engineer_features(output)
    if drop_columns is not None:
        output = output.drop(columns=drop_columns)
    return output