"""
Helper funtions  for filtering the image overview dataframe based on the
dataset versions and class mapping.
"""

import logging
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)


def prepare_versions_to_filter(version_input: Optional[str | list[str]]) -> list[str]:
    """Prepare the dataset version to be used for filtering.

    Args:
        version: Version of the dataset to be used.

    Returns:
        A list of the dataset versions to filter by. If "all" or
        None is given, an empty list is returned to indicate no filtering.

    Examples:
        - simple string: "v1" -> ["v1"]
        - list of strings: ["v1", "v2"] -> ["v1", "v2"]
        - "all" or None -> []
    """
    if version_input is None:
        logger.info("No dataset version provided, using all images.")
        return []

    if isinstance(version_input, str):
        if version_input == "all":
            logger.info("No dataset version provided, using all images.")
            return []

        logger.info("Dataset version provided: %s", version_input)
        return [version_input]

    if isinstance(version_input, list):
        logger.info("Dataset versions provided: %s", version_input)
        return version_input

    raise ValueError(
        f"Dataset version {version_input} could not be processed. {type(version_input)}"
    )


def filter_versions(version_list: list, df: pd.DataFrame) -> pd.DataFrame:
    """Filter the image overview dataframe based on the dataset versions provided.

    Args:
        df:  Dataframe to filter  with version columns as "version_" + version_str.

        version_list:  A List containing the dataset versions to filter by.
                        If a empty list is no version filtering will be applied.

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
        | img1.jpg   | True     |
    """
    logger.debug(
        "Filtering the dataframe based on the dataset versions: %s", version_list
    )
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

    return df.drop(df.filter(regex="version_").columns, axis=1)


def filter_class_mapping(
    class_map: dict, rest_classes: list[str] = [], priority_classes: list[str] = []
) -> pd.DataFrame:
    """Prepares the class map based on the provided rest and priority classes.

    To focus the training on specific classes, the class map is updated to set classes
    that are not in the priority classes to 0. The rest classes are to filter, wich classes
    should be kept in the class map alongside the priority classes. Empty rest and priority
    classes will result in no filtering.

    Args:
        class_map: Contains the class labels and their corresponding values.
        priority_classes : List of classes that keep their original mapping value.
                           If atleast one class is defined, all other classes are set to 0.
                           If empty, no priority classes are set.

        rest_classes: Classes to keep alonside the priority classes in the class map.
                        If empty, no classes are removed.


    Returns:
        A dictionary containing the updated class labels isnide the rest classes and
        priority classes.

    Examples:
        Given:

        .. code-block:: python
            priority_classes = ["class1"]
            rest_classes = ["class2"]

            class_map = {
                "class1": 1,
                "class2": 2,
                "class3": 3

        The resulting class map would be:

        ..code-block:: python
            {
                "class1": 1,
                "class3": 0
            }
    """

    logging.info(
            "Classes to keep based on defined priority classes, if emtpy no prio are set:%s \n \
            Classes to keep based on defined rest classes. If empty, no class are filtered out: %s",
        rest_classes,
        priority_classes,
    )

    return {
        key: (value if key in priority_classes or priority_classes == [] else 0)
        for key, value in class_map.items()
        if key in priority_classes or key in rest_classes
    }
