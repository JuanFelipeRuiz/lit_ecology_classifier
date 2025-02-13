"""
get_ecology_data.py
===================

Script to download and extract the main supported datasets for the lit_ecology_classifier project.
It downloads the datasets from the Eawag Open Data platform Eric and extracts them to the data folder.

Supported datasets:

- ZooLake1
- ZooLake2
- OOD
- Soon to be added: ZooLake3
- Soon to be added: Mini ZooLake2
- Soon to be added: Mini OOD

Example usage:

```bash / cmd
python -m lit_ecology_classifier.get_data ZooLake1
```
"""
import os
import argparse
import sys
import shutil
import zipfile
from pathlib import Path
import logging
import json

import requests

class GetEcologyData:
    def __init__(self, dataset):
        self.dataset = dataset
        self.url = None
        self.data_folder_path = Path("data")
        self.dataset_path = self.data_folder_path / self.dataset
        self.zip_file = None

    def get_dataset_urls(self):
        """Return the download URL for the specified dataset.

        Args:
            dataset: The name of the dataset.

        Returns:
            The URL to download the dataset.

        Raises:
            ValueError: If the dataset is not a valid option.

        """
        dataset_urls = {
            "mini_dataset": "only a few images for testing",
            "mini_OOD": "only a few images for testing",
            "ZooLake1": "https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip",
            "ZooLake2": "https://opendata.eawag.ch/dataset/1aee0a2a-e1a7-4910-8151-4fc67c15dc63/resource/e241f3df-24f5-492a-9d5c-c17cacab28f2/download/2562ce1c-5015-4599-9a4e-3a1a1026af50-zoolake2.zip",
            "OOD": "https://opendata.eawag.ch/dataset/1aee0a2a-e1a7-4910-8151-4fc67c15dc63/resource/0262dc4f-f165-41e5-923d-f8eb3e744f4f/download/48b71667-4d12-4396-aec9-b760977eeb72-ood_data.zip",
            # "ZooLake3": "https://example.com/dataset3.zip",
        }

        if self.dataset not in dataset_urls:
            raise ValueError(
                f"Dataset not available. Currently supported datasets: {dataset_urls.keys()}"
            )
        
        self.url = dataset_urls[self.dataset]

    def download_file(self):
        """Download a file from a given URL and save it to the specified destination.

        Raises:
            ValueError: If the download fails.
        """
        try:
            response = requests.get(self.url, stream=True, timeout=10)

            response.raise_for_status()

            with open(self.zip_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logging.info("Downloaded to %s", self.zip_file)

        except requests.RequestException as e:
            raise ValueError(f"Could not download file from {self.url}: {e}") from e
        
    def search_for_zip(self):
        """Search for a zip file in the current directory containing the dataset name"""
        self.zip_file = self.data_folder_path / f"{self.dataset}.zip"
        if self.zip_file.exists():
            print(f"Found an existing zip file for {self.dataset}.")
            response = input("Do you want to extract it? (y/n): ")
            if response.lower() == "y":
                self.zip_file = self.data_folder_path / f"{self.dataset}.zip"
                return True
            elif response.lower() == "n":
                print("ignoring the existing zip file.")

        return False

       

    def extract_zip(self):
        """Extract a zip file to the specified directory.

        Extracts the zip file to the data folder if there is no common prefix in the zip file.
        If there is a common prefix, it extracts the zip file to the parent folder  "data"
        and renames afterwards the parent folder to the dataset name.

        Raises:
            ValueError: If the zip file is invalid.

        """
        try:
            logging.info("Extracting the zip file: %s", self.zip_file)
            with zipfile.ZipFile(self.zip_file, "r") as zip_ref:
  
                common_prefix = os.path.commonprefix(zip_ref.namelist())

                if common_prefix == "":
                    zip_ref.extractall(self.dataset_path)
                    

                else:
                    zip_ref.extractall(self.data_folder_path)
                    # rename the parent folder to the dataset name
                    folder_to_rename = self.data_folder_path / common_prefix
                    folder_to_rename.rename(self.dataset_path)

                
            logging.info("Extracted to %s", self.dataset_path)
        except zipfile.BadZipFile as e:
            raise ValueError(f"could not extract the zip file: {e}") from e


    def prepare_folders(self):
        """Create a folder for the specified dataset

        If a folder with the same name already exists,
        it will ask the user if he wants to overwrite it.

        Args:
            dataset: The name of the dataset.

        """
        if self.dataset_path.exists() and  self.dataset_path.suffix != '.zip':
            print(f"The target folder {self.dataset_path} already exists.")
            response = input("Do you want to overwrite it? (y/n): ")
            if response.lower() != "y":
                print("Exiting...")
                sys.exit(0)

            shutil.rmtree(self.dataset_path)

    def set_conifg_file(self):
        """Append the dataset name to the config file."""

        config_path = Path(".") / "config" / "dataset_versions.json"

        if not config_path.exists():
            logging.info("Config file not found, creating a new one.")

            # check if the parent folder exists, if not create it
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # create a new config file with the dataset
            with open(config_path, "w") as file:
                json.dump({self.dataset: str(self.dataset_path)}, file)

        else:
            # open the existing config file
            with open(config_path, "r") as file:
                data = json.load(file)

            # transform the values to Path objects
            data = {key: Path(value) for key, value in data.items()}

            # check if the dataset is already in the config file with the same path and no other datasets are present
            if (
                self.dataset in data
                and data[self.dataset] == self.dataset_path
                and data.keys() == {self.dataset}
            ):
                return

            print(f"Config file found at {config_path} please choose an option:")
            response = input(
                "  - append the new dataset [0]\n  - overwrite the config file [1]\n  - exit without changes to the config file [2]\nUser input(0/1/2):"
            )

            if response == "0":
                logging.info(
                    "Appending %s to the config file while ensuring all paths works with the operating system.",
                    self.dataset,
                )

                data[self.dataset] = str(self.dataset_path)

                with open(config_path, "w") as file:
                    json.dump(data, file)

            elif response == "1":
                logging.info(
                    "Overwriting the config file with the new dataset %s.", self.dataset
                )
                with open(config_path, "w") as file:
                    json.dump({self.dataset: str(self.dataset_path)}, file)

            elif response == "2":
                print("No changes made to the config file.")
                return

            else:
                print("Invalid input, assuming exit without changes.")
                return

    def check_for_parent_folder_in_zip(self):
        """Check if there is a common parent folder in the zip file that could be replaced."""
        # extract the parent folder name from the zip file to rename it later
        with zipfile.ZipFile(self.zip_file, "r") as zip_ref:
            
            # check if there is a common prefix in the zip file that may be used as a extra folder
            common_prefix = os.path.commonprefix(zip_ref.namelist())
            print("common_prefix:", common_prefix)
            print("common datatype:", type(common_prefix))
            if common_prefix == "":
                return self.dataset
                

            return Path(common_prefix).parts[0]

        logging.info("Extracting the dataset: %s", dataset)



    def main(self):
        """Main function to download and extract the dataset."""
      

        logging.info("Preparing the folders for: %s", dataset)
        self.get_dataset_urls()

        self.prepare_folders()

        zip_found = self.search_for_zip()

        if not zip_found:
            logging.info("Downloading the dataset: %s", dataset)
            self.download_file()

        # extract the zip file
        self.extract_zip()

        # if the zip file was downloaded, remove it 
        if not zip_found:
            self.zip_file.unlink()
            logging.info("Dataset %s is ready at %s", self.dataset_path)

        self.set_conifg_file()

def argss():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and extract datasets.")
    parser.add_argument(
        "dataset", type=str, help="The name of the dataset to download."
    )
    return parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # start time of the script
    import time

    start_time = time.time()


    args = argss()
    dataset = args.dataset

    GetEcologyData(dataset).main()

    # end time of the script
    total_secs = time.time() - start_time
    print("Time taken for downloading the data (in secs): %s", total_secs)
