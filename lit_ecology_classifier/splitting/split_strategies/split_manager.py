"""
Manages the diffrent input types of split strategy for splitting of the image overview.
Can handle the following input types:
- a string with the class name of the wanted split strategy.
  # TODO: Add the possibility to import split strategies from other packages.
- a initialized instance of the BaseSplitStrategy class.
- a subclass of the BaseSplitStrategy class that needs to be initialized with the split_args.

If no split strategy is provided, a error is raised. The main function of the class is
the `perform_split` method that applies the split strategy on the given image overview.

More information can be found in the docstring of the modul `base_split_strategy.
"""

import inspect
import logging

import pandas as pd

from lit_ecology_classifier.helpers import helpers
from lit_ecology_classifier.splitting.split_strategies.base_split_strategy import (
    BaseSplitStrategy,
)

logger = logging.getLogger(__name__)


class SplitManager:
    """Manages the diffrent types of split strategy for splitting of the image overview.

    Uses the given split strategy to split the image overview into the defined splits.
    The main function of the class is the `perform_split` method that performs the
    split strategy.

    Attributes:
        split_strategy: Split strategy to be used for the splitting of the image overview.
        split_args: A optional dictionary containing the arguments to be used for the split
                    strategy.If a initialized instance of the split strategy is given,
                    the split_args will be ignored.
    """

    def __init__(
        self,
        split_strategy: str | BaseSplitStrategy,
        split_args: dict = None,
    ):
        self.split_strategy = self._initialize_split_strategy(
            split_strategy, split_args
        )

    def _initialize_split_strategy(
        self, split_strategy: str | BaseSplitStrategy, split_args: dict = None
    ) -> BaseSplitStrategy:
        """Handels the initialization of the split strategy based on the given input.

        Args:
            split_strategy: Can be one of the following:
                -  a string with the class name of the split strategy to be used
                -  an instance of the split strategy to be used
                -  a subclass of the BaseSplitStrategy class that needs to be
                    initialized with the GIVEN  split_args
                If no split strategy is provided a error is raised.
            split_args: A dictionary containing the arguments to be used for the
                    split strategy.
        """
        # initailize the base split strategy if no split strategy is provided
        if split_strategy is None:
            return BaseSplitStrategy()

        # returns the given split strategy if it is already an instance of SplitStrategy
        if isinstance(split_strategy, BaseSplitStrategy):
            return split_strategy

        # initialize the split strategy based on the given subclass of SplitStrategy
        if inspect.isclass(split_strategy) and issubclass(
            split_strategy, BaseSplitStrategy
        ):
            return split_strategy(**split_args)

        # initialize the split strategy based on the given string
        if isinstance(split_strategy, str):

            # import the class based on the given string

            imported_class = helpers.import_class(
                class_name=split_strategy,
                modul_path="lit_ecology_classifier.splitting.split_strategies.",
            )

            # intialize the split strategy without arguments if no split_args are given
            if split_args == {} or split_args is None:
                return imported_class()

            # initialize the imported split strategy with the given arguments
            return imported_class(**split_args)

        logger.error(
            "Split strategy has to be a string or an instance of SplitStrategy"
        )

        raise ValueError(
            "Split strategy has to be a string or an instance of SplitStrategy"
        )

    def _transform_split_dict(self, split_dict: dict) -> pd.DataFrame:
        """Transform the split dictionary into a dataframe with the columns image and split.

        Since the output of the split strategy is a dictionary with the split name as key,
        the function transforms the dictionary into a DataFrame with the columns `image`
        and `split`.

        Args:
            split_dict: A dictionary with the split name as key and the split data as value.

        Returns:
            A DataFrame containing the split data and further informations.

        Example:

            Given the following split dictionary:
                {
                    "train": img1, ...
                    "val": img2, ...
                    "test": img3,  ...
                }

            The function transforms the dictionary into the following DataFrame:

        |image|split|
        |:---:|:---:|
        |img1 |train|
        |img2 | val |
        |img3 | test|
        """
        # Create a combined DataFrame by iterating through the dictionary
        combined_df = pd.concat(
            [
                # Concatenate the image and target class to a DataFrame
                pd.concat([image, target_class], axis=1, names=["image", "y_label"])
                # assign the split name to the DataFrame
                .assign(split=split_name)
                # iterate through the dictionary
                for split_name, (image, target_class) in split_dict.items()
            ],
            ignore_index=True,
        )

        return combined_df

    def perfom_split(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Perform a split on the given image overview based on the split strategy.

        Args:
            filtered_df: The filtered image overview to be splitted

        Returns:
            A DataFrame containing the split data and further informations.
        """
        split_df = self.split_strategy.perform_split(filtered_df, y_col="class_map")
        return self._transform_split_dict(split_df)

    def get_untransformed_split(self, filtered_df) -> pd.DataFrame:
        """Gets the untrensformed split data from the split strategy.

        Args:
            split_dict: A dictionary with the split name as key and the split data as value.

        Returns:
            A DataFrame containing the split data and further informations.
        """
        return self.split_strategy.perform_split(filtered_df, y_col="class_map")
