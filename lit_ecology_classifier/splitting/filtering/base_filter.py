"""
Provides a abstract base class for the creation of custom filters to
filter the image overview before splitting the data into train and test sets.
It allows Users to implement their own filters by subclassing the `BaseImageFilter`
class ensuring that the needed methods are implemented.

To work properly, the custom filter has to be saved in a file in the `filtering` module.
The class name has to be in CamelCase and the file the exactly same name in snake_case.
If the created class is given uninitialized to the custom filter, the SplitProcessor will
initialize the filter with the given arguments. This would ensure the correctsaving of 
arguments inside the split overview.

Usage:
    - Define a custom filter by subclassing `BaseImageFilter`.
    - Implement the `filter_image_overview` method in the subclass.
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
    