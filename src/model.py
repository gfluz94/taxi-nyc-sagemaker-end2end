from typing import List, Tuple
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import GradientBoostingRegressor


SEED = 99
TARGET_COLUMN = ["fare_amount"]
CONTINUOUS_FEATURES_LOG = [
    "distance_ride"
]
CONTINUOUS_FEATURES = [
    "pickup_latitude", "pickup_longitude", 
    "dropoff_latitude", "dropoff_longitude"
]
INTEGER_FEATURES = ["passenger_count"]
CATEGORICAL_FEATURES = [
    "period_ride", "hour_day_ride"
]


class ColumnSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns: List[str]):
        """Custom sklearn transformer to select specific columns in a pandas dataframe.

        Attributes:
            columns (list of str) representing the names of columns to be selected
                
        """
        self.__columns = columns
        self.__df = pd.DataFrame
        
    @property
    def columns(self):
        return self.__columns
    
    def fit(self, X, y=None):
        assert isinstance(X, self.__df), \
            "X must be a pandas DataFrame"
        return self
    
    def transform(self, X):
        assert isinstance(X, self.__df), \
            "X must be a pandas DataFrame"
        return X.loc[:, self.__columns]


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, apply_log: bool = False, scaling: str = "min_max"):
        """Custom sklearn transformer to perform transformations to numerical features.

        Attributes:
            apply_log (bool) representing whether or not logarithm should be applied
            scaling (str) representing the scaling method (either `min_max` or `standard`)
                
        """
        self.__apply_log = apply_log
        assert scaling in ["min_max", "standard"], \
            "`scaling` should be either 'min_max' or 'standard'"
        self.__scaling = scaling
        if self.__scaling == "min_max":
            self.__scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.__scaler = StandardScaler()
        self.__log = np.log
        self.__exp = np.exp
        
    @property
    def scaling(self):
        return self.__scaling
    
    @property
    def apply_log(self):
        return self.__apply_log
    
    def fit(self, X, y=None):
        X_ = X.copy()
        if self.__apply_log:
            X_ = self.__log(X_)
        self.__scaler.fit(X_, y)
        return self
    
    def transform(self, X):
        X_ = X.copy()
        if self.__apply_log:
            X_ = self.__log(X_)
        return self.__scaler.transform(X_)
    
    def inverse_transform(self, X):
        X_ = X.copy()
        X_ = self.__scaler.inverse_transform(X_)
        if self.__apply_log:
            X_ = self.__exp(X_)
        return X_


def define_preprocessing_pipeline(continuous_features_log: List[str],
                                  continuous_features: List[str],
                                  integer_features: List[str],
                                  categorical_features: List[str]) -> Pipeline:
    """
    Function that builds the preprocessing pipeline.
    
    Args: 
        continuous_features_log (list of str): numerical columns to apply logarithm and normalization
        continuous_features (list of str): numerical columns to apply normalization
        integer_features (list of str): integer columns to apply min-max scaling
        categorical_features (list of str): categorical columns to one-hot encode
        
    Returns: 
        Pipeline object to preprocess the input features.
    """
    
    continuous_log_preprocessing = Pipeline(steps=[
        ("continuous_log_feats_selector", ColumnSelector(columns=continuous_features_log)),
        ("log_scaling", NumericalPreprocessor(apply_log=True, scaling="standard"))
    ])
    continuous_preprocessing = Pipeline(steps=[
        ("continuous_feats_selector", ColumnSelector(columns=continuous_features)),
        ("scaling", NumericalPreprocessor(apply_log=False, scaling="standard"))
    ])
    integer_preprocessing = Pipeline(steps=[
        ("integer_feats_selector", ColumnSelector(columns=integer_features)),
        ("scaling", NumericalPreprocessor(apply_log=False, scaling="min_max"))
    ])
    categorical_preprocessing = Pipeline(steps=[
        ("categorical_feats_selector", ColumnSelector(columns=categorical_features)),
        ("oh_encoding", OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])
    
    preprocessing = FeatureUnion(transformer_list=[
        ("continuous_log_preprocessing", continuous_log_preprocessing),
        ("continuous_preprocessing", continuous_preprocessing),
        ("integer_preprocessing", integer_preprocessing),
        ("categorical_preprocessing", categorical_preprocessing),
    ])
    
    return preprocessing


def define_candidate_model() -> Tuple[str, BaseEstimator, dict]:
    """
    Function that returns candidate models.
    
    Args: 
        None
        
    Returns: 
        List of tuples containing: model name, model object itself and parameters search hyperspace.
    """
    model_name = "gradient_boosting"
    gbc = GradientBoostingRegressor(random_state=SEED)
    gbc_params = {
        f"{model_name}__n_estimators": np.arange(80, 130, 20),
        f"{model_name}__max_depth": np.arange(4, 8, 1),
        f"{model_name}__learning_rate": np.logspace(-2, -1, 3)
    }
    
    return (model_name, gbc, gbc_params)


def define_preprocessing_pipeline(continuous_features_log: List[str],
                                  continuous_features: List[str],
                                  integer_features: List[str],
                                  categorical_features: List[str]) -> Pipeline:
    """
    Function that builds the preprocessing pipeline.
    
    Args: 
        continuous_features_log (list of str): numerical columns to apply logarithm and normalization
        continuous_features (list of str): numerical columns to apply normalization
        integer_features (list of str): integer columns to apply min-max scaling
        categorical_features (list of str): categorical columns to one-hot encode
        
    Returns: 
        Pipeline object to preprocess the input features.
    """
    
    continuous_log_preprocessing = Pipeline(steps=[
        ("continuous_log_feats_selector", ColumnSelector(columns=continuous_features_log)),
        ("log_scaling", NumericalPreprocessor(apply_log=True, scaling="standard"))
    ])
    continuous_preprocessing = Pipeline(steps=[
        ("continuous_feats_selector", ColumnSelector(columns=continuous_features)),
        ("scaling", NumericalPreprocessor(apply_log=False, scaling="standard"))
    ])
    integer_preprocessing = Pipeline(steps=[
        ("integer_feats_selector", ColumnSelector(columns=integer_features)),
        ("scaling", NumericalPreprocessor(apply_log=False, scaling="min_max"))
    ])
    categorical_preprocessing = Pipeline(steps=[
        ("categorical_feats_selector", ColumnSelector(columns=categorical_features)),
        ("oh_encoding", OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])
    
    preprocessing = FeatureUnion(transformer_list=[
        ("continuous_log_preprocessing", continuous_log_preprocessing),
        ("continuous_preprocessing", continuous_preprocessing),
        ("integer_preprocessing", integer_preprocessing),
        ("categorical_preprocessing", categorical_preprocessing),
    ])
    
    return preprocessing