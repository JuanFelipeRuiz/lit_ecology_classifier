"""
Abstract class to filter the image overview before splitting the 
data into train and test sets.
"""

from abc import ABC, abstractmethod

import pandas as pd

class BaseFilter(ABC):

    @abstractmethod
    def filter_image_overview(self, image_overview : pd.DataFrame):
        raise NotImplementedError("Must override filter logic")
    
    @abstractmethod
    def search_existing_splits(self, data):
        raise NotImplementedError(
            "Must override logic to search for existingsplits, since its based on the \
            used arguments for filtering"
            )
    