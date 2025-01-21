"""
Helper funtions  for filtering the image overview dataframe based on the
dataset versions and class mapping.
"""

import logging
from typing import Union

import pandas as pd


logger = logging.getLogger(__name__)


def prepare_args_to_filter(arg_input: Union[str, list[str], None]) -> list[str]:
    """Prepare a arg input to be used for filtering the image overview dataframe.

    Args:
        arg_input: A string or list of strings containing the column suffic to filter by.
                    If "all" or None is given, an empty list is returned to indicate
                    no filtering.

    Returns:
        A list of the given arg_input to filter the image overview dataframe.

    Examples:
        - simple string: "v1" -> ["v1"]
        - list of strings: ["v1", "v2"] -> ["v1", "v2"]
        - "all" or None -> []
    """
    if arg_input is None:
        return []

    if isinstance(arg_input, str):
        if arg_input == "all":
            return []

        return [arg_input]

    if isinstance(arg_input, list):
        return arg_input

    raise ValueError(f"{arg_input} could not be processed.")


def filter_versions(df: pd.DataFrame, version_list: list[str]) -> pd.DataFrame:
    """Filter the image overview to include images that are part of the given dataset version.

    Args:
        df:  Dataframe to filter  with version columns as "version_" + version_str.

        version_list:  A List containing the dataset versions to include in the dataset.
                          If empty, the original dataframe is returned.

    Returns:
        Dataframe fitlered by the dataset versions.

    Examples:
        Using the following dataframe:

        | iamge      |class    | version_v1 | version_v2 |
        |------------|---------|------------|------------|
        | img1.jpg   | class1  | True       | False      |
        | img2.jpg   | class1  | False      | True       |
        | img3.jpg   | class1  | False      | False      |
        | img4.jpg   | class1  | True       | True       |


        And the version_list: $ ["v1"], the resulting dataframe would be:

        | iamge      | class    |
        |------------|----------|
        | img1.jpg   | class1   |
    """
    logger.debug(
        "Filtering the dataframe based on the dataset versions: %s", version_list
    )
    # if version list is empty, use empty()
    if version_list == []:
        logger.debug("No dataset version provided, returning the original dataframe")
        return df

    version_col_name = ["version_" + version_str for version_str in version_list]

    if not any(col in df.columns for col in version_col_name):
        raise ValueError(
            f"Version columns {version_col_name} not found in the dataframe."
        )

    version_filter = df[version_col_name].any(axis=1)

    df = df[version_filter]

    return df.drop(df.filter(regex="version_").columns, axis=1).reset_index(drop=True)


def filter_ood_images(df: pd.DataFrame, ood_list: list[str]) -> pd.DataFrame:
    """Filters the given OOD images from the image overview dataframe out of the dataset.

    Out-of-Dataset (OOD) are images that are not part of the train, validation or test set.
    This were introduced by the research of cheng chen in the paper "Producing Plankton
    Classifiers that are Robust to Dataset Shift".

    Args:
        df: Dataframe to filter with the ood column.
        ood: A string to filter the dataframe by the ood column.
             If None, no filtering is applied.

    Returns:
        Dataframe filtered by the ood column.
    """

    if ood_list == [] or ood_list is None:
        logger.debug("No OOD defined, not filterig based on the OOD images.")
        return df

    logger.debug("Filtering the dataframe based on the OOD images: %s", ood_list)
    ood_col_name = ["OOD_" + version_str for version_str in ood_list]

    if not any(col in df.columns for col in ood_col_name):
        raise ValueError(f"OOD columns {ood_col_name} not found in the dataframe.")

    ood_filter = df[ood_list].any(axis=1)

    df = df[~ood_filter]

    return df.drop(df.filter(regex="OOD_").columns, axis=1)
