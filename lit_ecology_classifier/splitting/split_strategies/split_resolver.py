
import logging
import importlib

import pandas as pd


logger = logging.getLogger(__name__)

def perform_splitting(
    split_strategy_function: str,
    image_overview: pd.DataFrame,
    split_args: dict,
    module_path: str = "lit_ecology_classifier.splitting.split_strategies"
) -> pd.DataFrame:
    """
    Load and apply a split function by name from a given module path.

    Args:
        split_name: Name of the split function (e.g. "plankifier_split")
        image_overview: DataFrame to split
        split_args: Arguments to pass to the split function
        module_path: Module path where the split function lives

    Returns:
        splited image overview DataFrame
    """
    try:
        module = importlib.import_module(f"{module_path}.{split_strategy_function}")
        split_fn = getattr(module, split_strategy_function)

        logger.debug(f"Applying split: {split_strategy_function} with args {split_args}")
        return split_fn(image_overview, **split_args)

    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_path}.{split_strategy_function}' not found.")
    except AttributeError:
        raise ImportError(f"Function '{split_strategy_function}' not found in module '{module_path}.{split_strategy_function}'.")