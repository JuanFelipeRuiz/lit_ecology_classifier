import pandas as pd
from typing import Union
import lit_ecology_classifier.helpers.filter as filter_helpers

def plankton_filter(
    image_overview: pd.DataFrame,
    dataset_version: Union[str, list[str], None] = None,
    ood: Union[str, list[str], None] = None,
    rest_classes: Union[str, list[str], None] = None,
    **kwargs: Union[str, list[str], None]
) -> pd.DataFrame:
    """
    Filters the image overview for images that are part of the plankifier dataset.

    Args:
        image_overview: A DataFrame containing the image overview.
        dataset_version: Versions to include; if 'all' or None, includes all.
        ood: If specified, filters out OOD images with that tag.

    Returns:
        A filtered image overview dataframe.
    """
    dataset_version = filter_helpers.prepare_args_to_filter(dataset_version)
    ood = filter_helpers.prepare_args_to_filter(ood)

    df = filter_helpers.filter_versions(image_overview, dataset_version)
    df = filter_helpers.filter_ood_images(df, ood)

    # remove rest classes if specified
    if rest_classes:
        rest_classes = filter_helpers.prepare_args_to_filter(rest_classes)
        df = df[~df['class'].isin(rest_classes)]

    return df


def PlanktonFilter(
    image_overview: pd.DataFrame,
    dataset_version: Union[str, list[str], None] = None,
    ood: Union[str, list[str], None] = None
) -> pd.DataFrame:
    """
    Filters the image overview for images that are part of the plankifier dataset.

    Args:
        image_overview: A DataFrame containing the image overview.
        dataset_version: Versions to include; if 'all' or None, includes all.
        ood: If specified, filters out OOD images with that tag.

    Returns:
        A filtered image overview dataframe.
    """
    dataset_version = filter_helpers.prepare_args_to_filter(dataset_version)
    ood = filter_helpers.prepare_args_to_filter(ood)

    df = filter_helpers.filter_versions(image_overview, dataset_version)
    df = filter_helpers.filter_ood_images(df, ood)

    return df