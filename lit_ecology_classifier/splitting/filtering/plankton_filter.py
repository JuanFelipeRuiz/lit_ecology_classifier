"""
Custom filter for the plankifier dataset that inherited from the `BaseFilter` class.

The PlanktonFilter class is used to filter the image overview over the different versions of
the dataset and the OOD images introduced by the (research)[[https://arxiv.org/abs/2401.14256]
of cheng chen.  
"""

import logging

import pandas as pd

import lit_ecology_classifier.helpers.filter as filter_helpers
from lit_ecology_classifier.splitting.filtering.base_filter import BaseFilter

logger = logging.getLogger(__name__)


class PlanktonFilter(BaseFilter):
    """
    Filter to include and exclude images from the plankifier dataset based on the OOD
    and the dataset version.


    Assumes that version and ood columns are present in the image overview dataframe 
    with the following naming convention: "version_str" or "ood_str"
    The "str" part is the dataset version or the OOD name to filter by.

    Attributes:
        dataset_version: Dataset versions to include in the dataset. If "all" or None,
            all versions are included.
        ood: If given, it filters the images used for the OOD version out of the dataset.
    """

    def __init__(
        self, dataset_version: str | list[str] = None, ood: str | list[str] = None
    ):
        """
        Initializes the PlankifierVersionFilter.

        Args:
            dataset_version:
                Dataset versions to include in the dataset. If "all" or None,
                all versions are included.
            ood:
                If given, it filters the images used for the OOD out of the dataset.
        """
        self.dataset_version = filter_helpers.prepare_args_to_filter(dataset_version)
        self.ood = filter_helpers.prepare_args_to_filter(ood)

    def filter_image_overview(self, image_overview: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the image overview for images that are part of the plankifier version.

        Args:
            image_overview: A DataFrame containing the image overview.

        Returns:
            The filtered image overview dataframe.
        """
        filterd_image_overview = filter_helpers.filter_versions(
            image_overview, self.dataset_version
        )

        filterd_image_overview = filter_helpers.filter_ood_images(
            filterd_image_overview, self.ood
        )
        return filterd_image_overview
