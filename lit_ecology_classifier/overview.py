"""
Generate Overview Script for the ZooLake Dataset
================================================

Generates an overview of images in the given dataset paths and saves the results
as a CSV file inside a dataset-specific artifacts folder. It is tailored for the ZooLake dataset 
and the images of the Aquascope project. 

To trigger the right overview creation, please refer the different versions as OOD, ZooLake1, ZooLake2, etc.


Parameters
----------
--dataset_versions : str
    Path to a JSON file containing the dataset version paths.
--dataset : str
    Name of the dataset or project to create the overview for. Defines the name of the output folder.
--overview_filename : str, optional
    Name of the output file to save the overview DataFrame. Defaults to "overview.csv".

Example
-------
First, create a JSON file (e.g., ``config/dataset_versions.json``) with following structure:

```json
{
    "1": "path/to/dataset/version1",
    "2": "path/to/dataset/version2"
}
```

Then run the script::

```bash
python -m lit_ecology_classifier.overview \
        --dataset_versions config/dataset_versions.json \
        --dataset xy \
        --overview_filename overview.csv
``` 
This will save the overview to ``data/xy_artefacts/overview.csv``.
"""

import logging
import pathlib
import os
import sys

from time import time

from lit_ecology_classifier.data_overview.zoolake_overview import ZooLakeOverviewCreator
from lit_ecology_classifier.helpers.argparser import overview_argparser

# Start timing the script
time_begin = time()

if __name__ == "__main__":
    # Print the script name and arguments
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format="%(filename)s - %(levelname)s -%(message)s", force=True)
    logger = logging.getLogger(__name__)


    print("\nRunning", sys.argv[0], sys.argv[1:])

    # Parse Arguments for creating the overview
    parser = overview_argparser()
    args = parser.parse_args()
    logger.info(args)

    # create the overview 
    overview_creator = ZooLakeOverviewCreator(dataset_version_paths= args.dataset_version_path_dict)
    df = overview_creator.get_overview_df()

    # create the output folder and ensure it exists
    output_folder = pathlib.Path("data") / f"{args.dataset}_artefacts" 
    output_folder.mkdir(parents=True, exist_ok=True)

    # create the output file path and remove it if it already exists
    output = pathlib.Path(output_folder, args.overview_filename)
    if os.path.exists(output):
        os.remove(output)

    df.to_csv(output, index=False)

    # create a gitignore for the new folder
    with open(pathlib.Path(output_folder,".gitignore"), "w") as gitignore_file:
        gitignore_file.write("*")

        
    logging.info("Saved overview to %s.", output)
    logging.info("Total time taken: %s seconds", time()-time_begin)
