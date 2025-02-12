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

import argparse
import sys
import shutil
import requests
import zipfile
from pathlib import Path
import logging

def get_dataset_urls(dataset : str):
    """Return the download URL for the specified dataset.
    
    Args:
        dataset: The name of the dataset.

    Returns:
        The URL to download the dataset.

    Raises:
        ValueError: If the dataset is not a valid option. 
    
    """

    dataset_urls = {
        "ZooLake1": "https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip",
        "ZooLake2": "https://opendata.eawag.ch/dataset/1aee0a2a-e1a7-4910-8151-4fc67c15dc63/resource/e241f3df-24f5-492a-9d5c-c17cacab28f2/download/2562ce1c-5015-4599-9a4e-3a1a1026af50-zoolake2.zip",
        "OOD": "https://opendata.eawag.ch/dataset/1aee0a2a-e1a7-4910-8151-4fc67c15dc63/resource/0262dc4f-f165-41e5-923d-f8eb3e744f4f/download/48b71667-4d12-4396-aec9-b760977eeb72-ood_data.zip"
        #"ZooLake3": "https://example.com/dataset3.zip",
    }
    
    if dataset not in dataset_urls:
        raise ValueError("Dataset not available. Currently supported datasets:%s", dataset_urls.keys())
    
    return dataset_urls[dataset]



def download_file(url : str, dest_path):
    """Download a file from a given URL and save it to the specified destination.

    Args:
        url: The URL of the file to download.
        dest_path: The destination path where the file will be saved.
    
    Raises:
        ValueError: If the download fails.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logging.info(f"Downloaded to {dest_path}")

    except requests.RequestException as e:
        raise ValueError(f"Could not download file from {url}: {e}") from e

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory.
    
    Raises:
        ValueError: If the zip file is invalid.
    
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info("Extracted to %s", extract_to)
    except zipfile.BadZipFile as e:
        raise ValueError("could not extract the zip file: %s", e) from e


def prepare_folders(dataset : str):
    """Create a folder for the specified dataset 
    
    If a folder with the same name already exists, 
    it will ask the user if he wants to overwrite it.

    Args:
        dataset: The name of the dataset.

    Returns:
        The path to the folder where the dataset will be saved.
    """

    data_folder = Path.cwd() / "data" 
    dataset_folder = data_folder / dataset

    if dataset_folder.exists():
        print(f"The folder {dataset_folder} already exists.")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != "y":
            print("Exiting...")
            sys.exit(0)
        
        shutil.rmtree(dataset_folder)

    data_folder.mkdir(parents=True, exist_ok=True)
    return data_folder


def argss():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and extract datasets.")
    parser.add_argument("dataset", type=str, help="The name of the dataset to download.")
    return parser.parse_args()


def main():
    """ Main function to download and extract the dataset."""
    args = argss()
    dataset = args.dataset


    logging.info(f"Preparing the folders for: {dataset}")
    url = get_dataset_urls(dataset)
    data_folder = prepare_folders(dataset)
    

    logging.info(f"Downloading the dataset: {dataset}")
    temp_file = data_folder / f"temp_{dataset}.zip"
    download_file(url, temp_file)

    # extract the parent folder name from the zip file to rename it later
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        parent_folder = zip_ref.namelist()[0].split("/")[0]
    
    logging.info(f"Extracting the dataset: {dataset}")
    extract_zip(temp_file, data_folder )

    # rename the parent folder to the dataset name  
    extracted_folder = data_folder  / parent_folder
    extracted_folder.rename(data_folder / dataset)
    
    temp_file.unlink()
    logging.info(f"Dataset {dataset} is ready at {data_folder / dataset}")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # start time of the script
    import time
    start_time = time.time()

    main()

    # end time of the script
    total_secs = time.time() - start_time
    print("Time taken for downloading the data (in secs): {}".format(total_secs))
