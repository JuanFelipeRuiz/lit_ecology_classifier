"""
Provides logic to reload a split from the split overview.
"""

import os
import pandas as pd
import logging

from lit_ecology_classifier.helpers.hashing import HashGenerator

logger = logging.getLogger(__name__)


def reload_logic(processor, row: pd.DataFrame):
    """
    Core logic to reload a split and restore args from split_overview row.
    """
    row = row.iloc[0]

    try:
        hash_value = row["combined_split_hash"]
    except KeyError:
        logger.error("Row is missing 'combined_split_hash': %s", row)
        raise KeyError()

    filepath = prepare_split_path(processor, hash_value)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Split file not found at {filepath}")

    reconstruct_args_from_row(processor, row)

    reloaded_df = pd.read_csv(filepath, index_col=0)


    regenerated = processor._generate_split_hash(reloaded_df)
    if regenerated != hash_value:
        logger.warning(
            f"Hash mismatch: {regenerated} != {hash_value}. "
            "This may indicate the split was modified after creation."
        )

    logger.info(f"Reloaded split from: {filepath}")
    return reloaded_df 



def reconstruct_args_from_row(processor, row: pd.Series):
    """
    Reconstructs split_args and filter_args on the processor from a row.
    """
    processor.split_args = {k[6:]: row[k] for k in row.index if k.startswith("split_")}
    processor.filter_args = {k[7:]: row[k] for k in row.index if k.startswith("filter_")}


def prepare_split_path(processor, hash_value: str) -> str:
    """
    Builds full path to the split file based on hash and artefact folder.
    """
    filename = f"{hash_value[:24]}.csv"
    return os.path.join(processor.artefact_folder, "splits", filename)
