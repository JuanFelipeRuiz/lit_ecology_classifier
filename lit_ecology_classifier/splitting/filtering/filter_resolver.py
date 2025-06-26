
import logging
import importlib

import pandas as pd


logger = logging.getLogger(__name__)

def apply_filter_by_name(
    filter_name: str,
    image_overview: pd.DataFrame,
    filter_args: dict,
    module_path: str = "lit_ecology_classifier.splitting.filtering"
) -> pd.DataFrame:
    """
    Load and apply a filter function by name from a given module path.

    Args:
        filter_name: Name of the filter function (e.g. "plankifier_version_filter")
        image_overview: DataFrame to filter
        filter_args: Arguments to pass to the filter function
        module_path: Module path where the filter function lives

    Returns:
        Filtered image overview DataFrame
    """
    try:
        module = importlib.import_module(f"{module_path}.{filter_name}")
        filter_fn = getattr(module, filter_name)

        logger.debug(f"Applying filter: {filter_name} with args {filter_args}")
        return filter_fn(image_overview, **filter_args)

    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_path}.{filter_name}' not found.")
    except AttributeError:
        raise ImportError(f"Function '{filter_name}' not found in module '{module_path}.{filter_name}'.")