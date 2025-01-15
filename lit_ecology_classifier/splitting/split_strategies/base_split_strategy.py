"""
Provides a abstract base class for the creation of custom split 
strategies. It allows Users to implement their own split strategies by
subclassing the `BaseSplitStrategy` class ensuring that the needed methods
are implemented.

To work properly, the custom filter has to be saved in a file in the 
`split_strategies` module. The class name has to be in CamelCase and the file
the exactly same name in snake_case.

If the created class is given uninitialized to the custom filter, the SplitProcessor
will initialize the filter with the given arguments. This would ensure the correct
saving of arguments inside the split overview.

Usage:
    - Define a custom filter by subclassing `BaseSplitStrategy`.
    - Implement the `perform_split` method in the subclass.
    - Instantiate the custom filter and pass it to the `Splitter` class.
    - To save the custom filter, add it to the `split_strategies` module.
        and name the file with the name of the custom filter in snake_case.
        

Example:
    class MyOwnSplitter(BaseSplitStrategy):
        def perform_split(self, data, y_col="class"):
            # Split logic here

            return {
                "train": [X_train, y_train],
                "val": [X_val, y_val],
                "test": [X_test,y_test]
                "any_other_split": [X_any_other_split, y_any_other_split]
            }

# Optional: Save the custom filter in a file named my_own_filter.py
"""

from abc import ABC, abstractmethod

class BaseSplitStrategy(ABC):
    """ Base class to define the interface for custom split strategies.

    The abstract class is used to specify that the `perform_split` method
    should be implemented by the development of custom split strategies.
    """

    @abstractmethod
    def perform_split(self, data, y_col = "class" ) -> dict[str, list]:
        """ Perform a split on the data.

        Args: 
            data: data  containing the image names and the class labels
            y_col: The column name of the target variable if data is a DataFrame
        """
        raise NotImplementedError("Must override perform_split")
