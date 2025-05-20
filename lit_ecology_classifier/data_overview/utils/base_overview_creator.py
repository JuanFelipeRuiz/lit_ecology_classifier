import concurrent.futures
import itertools
import logging
import os
from pathlib import Path
import warnings
from typing import Union
       


import pandas as pd

from lit_ecology_classifier.checks.duplicates import check_duplicates
from lit_ecology_classifier.data_overview.utils.image_processing import ProcessImage
from lit_ecology_classifier.data_overview.utils.raw_split_preparer import _RawSplitPathPreparer
from lit_ecology_classifier.data_overview.utils.raw_split_applier import _RawSplitApplier


class BaseOverviewCreator:
    """Base class for creating an overview of the images in the dataset versions

    This class is designed to be subclassed and provides the basic functionality and orchestration for creating an overview of the images in the dataset versions.
    It includes methods for processing images for a raw dataset, preparations steps before cleaning the dataset, and methods for creating an overview DataFrame with
    image metadata and hashes.

    The steps involved in creating the overview are as follows:

    - For the raw dataset:
        1. Collect image paths from the dataset versions
        2. Extract the needed metadata from the image paths based on the version of the dataset

    - Afterwards for the overview dataset:
        1. Check for duplicates in the dataset
        2. Attach additional features to the dataset, that are not part of the
          image metadata but needed for the cleaning step or further processing-
        3. Clean up the raw dataset (needs to be implemented in subclasses)
        4. Checks for the dataset (needs to be implemented in subclasses)
    """

    process_image = ProcessImage

    def __init__(self, dataset_version_paths: dict = None, 
                 ImageProcessor = None):
        """Initialize the OverviewCreator with the given ZooLake dataset versions and hash algorithm

        Args:
            dataset_versions_path: A dictionary containing the paths to the different ZooLake dataset versions. 
                                        Defaults to None.
            ImageProcessor: A custom image processor class. Needs to implement the process_image method.
        """

        self.dataset_versions_path= self._check_dataset_paths(dataset_version_paths)

        self.image_paths = self._prepare_image_paths()

        # Initialize the image processor
        self.image_processor = ProcessImage() if ImageProcessor is None else ImageProcessor


        self.split_applier = None  
        self._images_list = []
        self._overview_df = None
        self._overview_with_splits_df = None
        self._duplicates_df = None

    def _check_dataset_paths(self, dataset_version_paths: dict) -> dict:
        """Checks if the given paths of the dataset versions exist and are valid

        Args:
            dataset_version_paths (dict): A dictionary containing the paths to the different ZooLake dataset versions

        Returns:
            The dataset_version_paths dictionary if all paths are valid

        Raises:
            FileNotFoundError: If any of the paths do not exist
        """
        # If no paths are provided, use the default paths
        if dataset_version_paths is None:
            dataset_version_paths = {
                "1": os.path.join("data", "raw", "data"),
                "2": os.path.join("data", "raw", "ZooLake2"),
            }

        # check if each path exists
        for path in dataset_version_paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path} does not exist. Current working directory: {os.getcwd()}")

        return dataset_version_paths

    def _collect_image_paths_from_folder(self, version_path) -> list[str]:
        """Collects image paths from the specified folder and its subfolders.

        Searches the specified folder and its subfolders recursively for image files with a `.jpeg` extension.
        It returns the full paths to each image found,  preparing the list of image paths for further processing.

        Args:
            version_path (str): Conaints version and path to the class. Example: "path/to/zoolakev1"

        Returns:
            list[str]: List of found image paths
        """

        image_path = [
            # join the root path with the file name
            os.path.join(root, file)
            # walk through the folder and subfoledrs (generates lists of filespath and filenames)
            for root, _, files in os.walk(version_path)
            # loop through the files in the folder
            for file in files
            # filter for files that end with .jpeg
            if file.endswith(".jpeg")
        ]

        return image_path

    def _prepare_image_paths(self) -> dict:
        """ Prepares a list of dictionaries containing the dataset version and the image paths

        Returns:
            dict : A dictionary containing the dataset version as key and the image paths as values
                    Example:
                    {
                        "1": ["path1", "path2"],
                        "2": ["path1", "path2"]
                    }
        """

        return dict(
            map(
                lambda version_path: (
                    version_path[0],
                    self._collect_image_paths_from_folder(version_path[1]),
                ),
                self.dataset_versions_path.items(),
            )
        )

    def _process_images_by_version(self) -> list[dict]:
        """Applies the image processing function to a list containg tuples of version and image paths

        Input of process_image: version, path
        Example of image_paths:
            [("v1","path1"), ("v1","path2"), ("v2","path1")

        Args:
            None

        Returns:
            list[dict]: List of dictionaries containing the image metadata and hashes for all images in the dataset versions
        """
        # create a list of tuples containing the version and the image paths 
        version_image_pairs = itertools.chain.from_iterable(
            map(
                lambda item: zip(itertools.repeat(item[0]), item[1]),
                self.image_paths.items(),
            )
        )


        # Applies the image processing function to each image path and version
        # Version is needed, since diffrent versions may have diffrent preparations steps
        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_images = list(
            executor.map(
                lambda version_image: self.image_processor.process_image(*version_image),
                version_image_pairs,
            )
            )

        # extend the list with the dictionary of the processed images
        self._images_list.extend(processed_images)

        return self._images_list
    
    def attach_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for diffrent features to be attached to the dataset
        This method can be overridden in subclasses to implement specific feature extraction logic.

        Args:
            The dataframe containing the raw image metadata and hashes

        Returns:
            None
        """
        return df

    def clean_up_raw_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for diffrent cleaning methods to be applied to the raw dataset
        This method can be overridden in subclasses to implement specific cleaning logic.

        Args:
            df: The dataframe containing the raw image metadata and hashes

        Returns:
            pd.DataFrame: The cleaned up DataFrame
        """        
        return df
    
    def check_overview_df(self):
        """
        Placeholder for diffrent checks to be applied to the overview dataset.
        As example, if the expected values are in the dataset. 

        Note:
            Duplicates are already checked in the get_duplicates_df method and before the cleaning step.
        """
        pass

    def get_duplicates_df(self):
        """Get duplicates dataframe"""
        if self._duplicates_df is None:
            df = self.get_raw_df()
            self._duplicates_df = check_duplicates(df, by_data_set_version=True)
        return self._duplicates_df

    def get_raw_df(self):
        """Get the raw DataFrame containing only the image metadata and hashes, without any further processing."""
        if not self._images_list:
            self._process_images_by_version()
        return pd.DataFrame(self._images_list)

    def get_overview_df(self):
        """Get the overview DataFrame grouped and the data set version as hot encoded columns."""
        if self._overview_df is None:
            df = self.get_raw_df()
            df = self.attach_additional_features(df)
            self._overview_df = self.clean_up_raw_dataset(df)
            self.check_overview_df()
        return self._overview_df
    
    def save_overview_df(self, output_path: Union[str, Path]):
        """Save the overview DataFrame to a CSV file.

        Args:
            output_path: The path to the output CSV file.

        """
        overview_df = self.get_overview_df()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        overview_df.to_csv(output_path, index=False)
        logging.info(f"Saved overview to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    overview_creator = BaseOverviewCreator()
    overview_df = overview_creator.get_overview_df()
    print(overview_df.head())