"""
Manages the handling and initialization of the filter strategy and applies the 
filter strategy to the image overview. with the method `apply_filter`.

Can handle the following input types for the filter strategy:

- a string with the class name of the filter strategy to be used
    currently only packages in the `lit_ecology_classifier.splitting.filtering` are supported
- a initialized instance of the BaseFilter class
- a subclass of the BaseFilter class that need to be initialized with the filter_args
- if no filter strategy is provided, a error is raised.

"""

import logging
import inspect

import pandas as pd

from lit_ecology_classifier.helpers import helpers
from lit_ecology_classifier.splitting.filtering.base_filter import BaseFilter

logger = logging.getLogger(__name__)


class _FilterManager:

    def __init__(
        self,
        filter_strategy: str | BaseFilter,
        filter_args: dict = None,
    ):

        self.filter_strategy = self._initialize_filter_strategy(
            filter_strategy, filter_args
        )

    def _initialize_filter_strategy(
        self, filter_strategy: str | BaseFilter, filter_args: dict = None
    ) -> BaseFilter:
        """Handels the initialization of the filter strategy based on the given input.

        Args:
            filter_strategy: Can be one of the following:
                - a string with the class name of the filter strategy to be used
                - an instance of the filter strategy to be initialized
                    with the filter_args
                - a initialized instance of the BaseFilter class
                - if no filter strategy is provided, a error is raised.

        Returns:
            A initialized filter modul to be used for the filtering of the image
            overview.

        Raises:
            ValueError: If the filter strategy is not a string or an instance
            of BaseFilter.
        """

        if filter_strategy is None:
            return BaseFilter()

        # returns the given filter strategy if it is already an initialized filter strategy
        if isinstance(filter_strategy, BaseFilter):
            logger.info(
                "Ensure logging of the args, if a record of the filter strategy is wished.\
                Alternatively, the filter strategy can be initialized with the filter_args."
            )
            return filter_strategy

        # check if the filter strategy is a subclass of BaseFilter and initialize it
        if inspect.isclass(filter_strategy) and issubclass(filter_strategy, BaseFilter):

            return filter_strategy(**filter_args)

        # initialize the filter strategy based on the given string
        if isinstance(filter_strategy, str):
            imported_class = helpers.import_class(
                class_name=filter_strategy,
                modul_path="lit_ecology_classifier.splitting.filtering.",
            )

            # transform the filter args to a dictionary if it is not already a dictionary
            if filter_args == {} or filter_args is None:
                return imported_class()

            return imported_class(**filter_args)

        logger.error("Filter strategy has to be a string or an instance of BaseFilter")

        raise ValueError(
            "Filter strategy has to be a string or an instance of BaseFilter"
        )

    def apply_filter(self, image_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter strategy to the image overview.

        Args:
            image_df: The Dataframe containing the image overview.

        Returns:
            The filtered image overview.
        """
        return self.filter_strategy.filter_image_overview(image_df)
