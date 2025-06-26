"""
This module provides functions to search for existing splits in the split overview. 


"""
import logging
from typing import Union

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def find_with_existing_hash(processor, hash_value: str) -> pd.DataFrame:
    """ Find the split overview row matching the given hash value.
    
    Args:
        processor: The SplitProcessor instance containing the attribute split overview.
        hash_value: The hash value to search for in the split overview.

    Returns:
        pd.DataFrame: A DataFrame containing the matching split overview row.
    """
    split_overview = processor.split_overview

    # Ensure the split overview is loaded and contains the necessary column
    if split_overview is None or "combined_split_hash" not in split_overview:
        raise ValueError("Split overview is not loaded or missing 'combined_split_hash'.")
    
    # search for the hash value in the split overview and return the matching row 
    match = split_overview[split_overview["combined_split_hash"] == hash_value]
    if match.empty:
        raise ValueError(f"No split found with the given hash: {hash_value}")
    return match.iloc[[0]]


def find_existing_split_entry(processor, columns_to_ignore: list = None) -> pd.DataFrame:
    """Search for an existing split in the split overview that matches the defined  args and strategies.
    
    To adjust the columns please use the hook `lit_ecology_classifier.splitting.split_search.search_for_existing_split`.
    
    Args:
        processor: The SplitProcessor instance containing the split overview and args.
        columns_to_ignore: Optional list of column names to ignore when searching for matches.

    Returns:
        A dataframe containing the matching split overview row, or an empty dataframe if no match is found.
    
    """
    split_overview = processor.split_overview
    if split_overview is None or split_overview.empty:
        return pd.DataFrame()


    # Prepare the args with prefixes for matching
    prefixed_args = {f"split_{k}": v for k, v in processor.split_args.items()}
    prefixed_args.update({f"filter_{k}": v for k, v in processor.filter_args.items()})
    prefixed_args["split_strategy"] = processor.split_strategy
    prefixed_args["filter_strategy"] = processor.filter_strategy

    logger.info(f"Searching for existing split with args: {split_overview}")
    # Apply the matching function to each row of the relevant columns
    matches = split_overview.apply(
        lambda row: find_matching_args(split_entry=row, new_entry_args=prefixed_args , columns_to_ignore=columns_to_ignore),
        axis=1
    )

    # Return the matching row. If multiple matches are found, raise an error.
    if matches.any():
        if matches.sum() > 1:
            raise ValueError("Multiple matching split rows found.")
        return split_overview.loc[matches].iloc[[0]]

    # Return an empty DataFrame if no matches are found
    return pd.DataFrame()


def find_matching_args(split_entry, new_entry_args, columns_to_ignore=None):
    for key, expected in new_entry_args.items():

        # Skip the columns that are specified to be ignored
        if columns_to_ignore is not None and key in columns_to_ignore:
            split_entry = split_entry.drop(key)  # Drop the key for the final check
            continue

        # If the key arg is not in the split entry, it means it is not present in the split overview and should not be considered a match
        if key not in split_entry:
            return False
        
        actual_raw = split_entry[key]
        # If the actual value is NaN, we treat it as an empty set
        if pd.isna(actual_raw) and pd.isna(expected):
            split_entry = split_entry.drop(key)  # Drop the key for the final check
            continue

        if normalize_value(actual_raw) != normalize_value(expected):
            return False
        
        # we drop the key from the split_entry to check in the end if all relevant keys were checked
        split_entry = split_entry.drop(key)

    # Check if all relevant keys were checked
    if not split_entry.empty:
        logger.warning("Since Some of the existing split entry keys were not checked: %s", split_entry.index.tolist())
        return False
    return True

def normalize_value(value):
    """Normalizes a value to a set of strings."""
    if value is None:
        return set()
    
    # Handle pandas Series/DataFrame
    if isinstance(value, (pd.Series, pd.DataFrame)):
        if value.empty:
            return set()
        value = value.iloc[0]  # Get the first element
    
    # Handle numpy array
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return set()
        value = value.tolist()
        
    # Handle numpy nan
    if isinstance(value, float) and np.isnan(value):
        return set()
        
    # Handle empty string
    if isinstance(value, str) and not value:
        return set()
        
    # Handle empty list/tuple
    if isinstance(value, (list, tuple)):
        if not value:
            return set()
        # Convert each element to string individually
        return set(str(x) for x in value)
    
    # Handle string with potential comma separation
    if isinstance(value, str):
        # Check if string looks like a list representation
        if value.startswith('[') and value.endswith(']'):
            # Remove brackets and split by comma
            cleaned = value[1:-1].replace("'", "").replace('"', "")
            return set(x.strip() for x in cleaned.split(',') if x.strip())
        return set(v.strip() for v in value.split(",") if v.strip())
    
    # Handle single values
    return {str(value)}
