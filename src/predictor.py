import os
import dill
from typing import List
import pandas as pd
from html import unescape

from model import get_predictions


class PredictorService(object):

    def __init__(self, model_path: str):
        """
        Trained model wrapper to predict instances during inferece serving time.
	
        Attributes:
            model_path (str) representing filepath to model artifact
                
        """
        self.__model_path = model_path

    @property
    def model_path(self):
        return self.__model_path

    def start(self):
        """
        Loading and instantiation of the trained model
        """
        if not hasattr(self, "__model"):
            self.__model, self.__post_processor = self.__get_model()

    def __get_model(self):
        with open(os.path.join(self.__model_path, "model.pkl"), "rb") as file:
            model = dill.load(file)

        with open(os.path.join(self.__model_path, "post_processor.pkl"), "rb") as file:
            post_processor = dill.load(file)

        return model, post_processor

    def predict(self, inputs: pd.DataFrame) -> List[float]:
        """
        Function that generates prediction based on raw input data
            
            Args: 
                inputs (pandas DataFrame): raw input in dataframe format

            Returns: 
                List of predicted probabilities
        """
        if not hasattr(self, "__model"):
            self.__model, self.__post_processor = self.__get_model()
        return get_predictions(
            model=self.__model,
            post_processor=self.__post_processor,
            inputs=inputs
        )