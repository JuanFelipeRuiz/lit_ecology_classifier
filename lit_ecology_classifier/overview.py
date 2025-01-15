"""
Script to create an overview of the given dataset versions and optionally summarise the dataset by copying all 
unique images to the output folder.

Steps to run the script:
1. create a dictionary inside a JSON File with the version as the key and the path to the dataset as the value. 
   Example:

   File path: "config/dataset_versions.json"
   daataset_versions.json:
    {
        "1": "path/to/dataset/version1",
        "2": "path/to/dataset/version2"
    }
    
2. Run the script with the following command:
    python overview.py --name xy  --image_version_path_dict "config/dataset_versions.json" --output "output" --summarise
"""
import logging
import pathlib
import os
import sys

from time import time

from lit_ecology_classifier.data_overview.overview_creator import OverviewCreator
from lit_ecology_classifier.data_overview.images_copier import ImageCopier
from lit_ecology_classifier.helpers.argparser import overview_argparser

# Start timing the script
time_begin = time()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - -%(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Print the script name and arguments
    print("\nRunning", sys.argv[0], sys.argv[1:])

    # Parse Arguments for creating the overview
    parser = overview_argparser()
    args = parser.parse_args()
    logger.info(args)


    # create the overview 
    overview_creator = OverviewCreator(zoolake_version_paths= args.image_version_path_dict)
    df = overview_creator.get_overview_df()

    # if the output folder is not the current directory, 
    # ensure the folder exists and raise an error if it does
    
    pathlib.Path(args.dataset).mkdir(parents=True, exist_ok=True)

    # prepare filename 

    output = pathlib.Path(args.dataset, args.overview_filename)

    # save overview to file and overwrite if it exist
    if os.path.exists(output):
        os.remove(output)

    df.to_csv(output, index=False)

    # summarise the overview by copying all uniqze images to the output folder
    if args.summarise_to:
        copier = ImageCopier(args.summarise_to, overview_creator)
        copier.copy_images()
        
    logging.info(f"Overview saved to {output}, total time taken: {time()-time_begin} seconds")
