"""
Script to split the data into training and testing sets. 
Can only use split or filter strategies that are predifined inside of
the lit_ecology_classifier package. To use custom strategies, follow the template   
provided in the filter or split_strategy base class and add the strategy to the
strategies directory. 

Example cmd:

python lit_ecology_classifier/split.py - --priority_classes 'config/priority.json' --rest_classes 'config/rest_classes.json' --dataet "ZooTestToDelete"
"""
import typing
import logging
import pathlib
import sys
import os
from time import time

import pandas as pd

from lit_ecology_classifier.splitting.lit_split_processor import SplitProcessor
from lit_ecology_classifier.helpers.argparser import split_argparser


logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(filename)s - %(lineno)d: %(message)s',
    force = True
)

logger = logging.getLogger(__name__)



def get_split(args: typing.Any) -> str:
    """Main script to get the splits"""
    split_processor = SplitProcessor(
                                    image_overview = image_overview_path,                                
                                    )
    filter_args = {
            "rest_classes":  args.rest_classes,
            "priority_classes":  args.priority_classes,
        }
    
    split_args = {}

    print(filter_args)
        # Append the  args to filter_args
        
    split_processor.run(
                        split_strategy = args.split_strategy,
                        filter_strategy =  args.filter_strategy,
                        split_args= split_args,
                        filter_args= filter_args,
        )
    split_processor.save_outputs(description= args.description)

    return split_processor.get_split_overview(), split_processor.get_class_map()



if __name__ == "__main__":
    print("\nRunning", sys.argv[0], sys.argv[1:])

    # Parse Arguments for training
    parser = split_argparser()
    
    args = parser.parse_args()
    
    logger.info(args)

    # prepare file and folder paths
    image_overview_path = "data/phyto_artifacts/overview.csv" #pathlib.Path(args.dataset)/args.overview_filename

    split_overview_path = pathlib.Path(args.dataset)/f"split_overview.csv"

    pathlib.Path(args.dataset).mkdir(parents=True, exist_ok=True)

    split_df, class_map = get_split(args)

 