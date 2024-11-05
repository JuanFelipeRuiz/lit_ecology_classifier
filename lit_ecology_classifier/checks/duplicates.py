"""
Modul providing the functionality to check for duplicates in the dataset
"""

import logging
import warnings

import pandas as pd


logging.basicConfig(level=logging.INFO)


def check_duplicates(df: pd.DataFrame, by_data_set_version=False) -> pd.DataFrame:
    """Check for duplicates in the given dataset based on the hash values

    The duplicates are identified by comparing the hash values of the images. 
    When a duplicate is found, a DataFrame is created with further information, 
    like if the duplicate images have the same class or image name.

    Args:
        df (pd.DataFrame)  : DataFrame containing the image metadata and hashes to
                            for duplicates. The DataFrame should contain the columns 
                            'sha256', 'image', and 'class'.

        by_data_set_version (bool): If True, the duplicates are grouped by
                            the hash code and the data_set_version. Default is False.              

    Returns:
        A new DataFrame containing the duplicates in the dataset based on hash values
    """
    
    required_columns = ["sha256", "image", "class"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "The DataFrame should contain the columns 'sha256', 'image', and 'class'"
        )

    columns_to_groupby = ["sha256"]

    # get column sha256 and data_set_version if they exist in a list for further processing
    if "data_set_version" in df.columns and by_data_set_version:
        columns_to_groupby.append("data_set_version")

    # save the duplicates hash values in a list
    duplicates = df[df.duplicated(subset=columns_to_groupby, keep=False)]

    if duplicates.empty:
        print("No duplicates found in the dataset")
        logging.info("No duplicates found in the dataset")
        return None

    group_counts = (
        duplicates.groupby(columns_to_groupby)
        .agg(
            # Count the number of duplicates
            count=("class", "size"),
            # Check if the class and image name are the same for all duplicates
            diffrent_class=("class", lambda x: x.nunique() != 1),
            diffrent_image_name=("image", lambda x: x.nunique() != 1),
        )
        .reset_index()
    )

    group_counts["count"] = group_counts["count"].astype(int)
   
    count_of_duplicates = group_counts["count"].sum()

    # merge image names based on the hash code and datalake version
    group_counts = pd.merge(
        group_counts,
        df[columns_to_groupby + ["image", "class"]],
        on=columns_to_groupby,
        how="left",
    )
    
    duplicates_df = group_counts[group_counts["count"] > 0]

    warnings.warn(f"Duplicates found in the dataset: {count_of_duplicates}")

    return duplicates_df
