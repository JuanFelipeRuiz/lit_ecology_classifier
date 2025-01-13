"""
Script to split the data into training and testing sets. 
Can only use split or filter strategies that are predifined inside of
the lit_ecology_classifier package. To use custom strategies, follow the template   
provided in the filter or split_strategy base class and add the strategy to the
strategies directory. 
"""

import logging
import pathlib
import sys
import os
from time import time

import pandas as pd

from lit_ecology_classifier.splitting.split_processor import SplitProcessor
from lit_ecology_classifier.helpers.argparser import split_argparser


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("\nRunning", sys.argv[0], sys.argv[1:])

    # Parse Arguments for training
    parser = split_argparser()
    
    args = parser.parse_args()
    
    logger.info(args)

    # prepare file and folder paths
    image_overview_path = pathlib.Path(args.dataset_name)/args.overview_filename

    split_folder_path = pathlib.Path(args.dataset_name)/"splits"
    split_overview_path = pathlib.Path(args.dataset_name) / "splits"/f"{args.dataset_name}_split_overview.csv"

    pathlib.Path(args.dataset_name).mkdir(parents=True, exist_ok=True)
    pathlib.Path(split_folder_path).mkdir(parents=True, exist_ok=True)

    if not split_overview_path.exists():
    
        new_entry = {
            "split_strategy": None,
            "filter_strategy": None,
            "combined_split_hash": None,
            "description": None,
            "class_map": None
        }

        pd.DataFrame(new_entry, index=[0]).to_csv(split_overview_path)
        

    split_processor = SplitProcessor(
                                split_overview = split_overview_path,
                                split_folder = split_folder_path,
                                image_overview= image_overview_path,
                                split_strategy= args.split_strategy,
                                filter_strategy=  args.filter_strategy
                                )