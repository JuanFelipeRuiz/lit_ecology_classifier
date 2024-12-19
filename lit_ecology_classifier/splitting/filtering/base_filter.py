"""
Provides a abstract base class for the creation of custom filters to
filter the image overview before splitting the data into train and test sets.
It allows Users to implement their own filters by subclassing the `BaseImageFilter`
class ensuring that the needed logic for the pipeline is implemented.

It is necessary that the class is named in CamelCase and the file 
has exactly the same name in snake_case.

Usage:
    - Define a custom filter by subclassing `BaseImageFilter`.
    - Implement the `filter_image_overview` method in the subclass.
    - Give the custom filter to the SplitProcessor to use it in the pipeline.
        Can be a initialized instance of the custom filter or the class instance
        with the filter arguments to be initialized inside of the SplitProcessor.
    - Optional: To save the custom filter, add it to the `filters` module and 
        name the file with the name of the custom filter in snake_case.

Example:
    class MyOwnFilter(BaseImageFilter):
        def filter_image_overview(self, image_overview):
            # Filter logic here

            return filtered_image_overview


    # Optional: Save the custom filter in a file named my_own_filter.py
    
"""

from abc import ABC, abstractmethod

import pandas as pd

class BaseFilter(ABC):
    """
    Filter the image overview before splitting the
    data into train and test sets.

    The abstract class is used to define `filter_image_overview` method
    should be implemented by the development of custom filters.
    (Keyword: Polymorphism)
    """
    @abstractmethod
    def filter_image_overview(self, image_overview: pd.DataFrame) -> pd.DataFrame:
        """Filters the image overview

        Args:
            image_overview: The image overview dataframe.

        Returns:
            A  filtered image overview dataframe
        """
        raise NotImplementedError("Must override filter logic")
    