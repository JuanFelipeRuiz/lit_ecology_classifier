"""
Class to read and add the split information to the DataFrame based on the paths provided in the split_file_paths dictionary.
"""
import os
import pickle

import numpy as np
import pandas as pd

class RawSplitApplier:
    """ Add columns that shows the corresponding split of the image perprovided  by the split list 
    
    load the image paths for the training, test, and validation splits  based on 
    the given split_file_paths dictionary.

    Attributes:
        _split_file_paths (dict): A dictionary containing the paths to the prepared splits for the
                                  different versions of the ZooLake dataset. Example:
                                    {"1": {
                                            "train": "path/to/train.txt", 
                                             "test": "path/to/test.txt", 
                                             "val": "path/to/val.txt"
                                            },
                                    "2": {"pickle": "path/to/split.pickle"},
    
        image_splits (dict): A dictionary containing the image paths for the training, test, and validation splits for each version of the ZooLake dataset.
    
    
    """

    def __init__(self, split_file_paths: dict):
        self._split_file_paths = split_file_paths
        self.image_splits = {}

    
    def _load_split_overview_from_pickle(self, version: str) -> dict:
        """Load the train/val and test image splits from a pickle file into a dictionary

        It loads the pickle file based on the prepared data path inside the split_file_paths dictionary and uses
        the given Version String as key, to get the path.

        It assumes that the pickle file contains a DataFrame with the image paths for the train, test, and validation splits.
        The order of the splits is based on Cheng Chen function SplitFromPickle inside [train_val_test_split.py  in the
        Plankiformer_OOD](https://github.com/cchen07/Plankiformer_OOD/blob/main/utils_analysis/train_val_test_split.py)
        repository.

        Args:
            version (str): Version of the ZooLake dataset that work as key for the path inside the split_file_paths dictionary

        Returns:
            dict: A dictionary containing the image paths for the train, test, and validation splits.
                    Example:
                    {
                        "train": [path/to/image1, path/to/image2, ...],
                        "test": [path/to/image3, path/to/image4, ...],
                        "val": [path/to/image5, path/to/image6, ...]
                    }

        Raises:
            ValueError: If an error occurs while unpickling the file
            Exception: If an unexpected error occurs while loading the pickle file
        """

        splits = ["train", "test", "val"]

        # Get the path to the pickle file from the prepared paths dictionary
        path_pickle_file = self._split_file_paths[version]["pickle"]

        try:
            # read the pickle file
            train_test_val_splits = pd.read_pickle(path_pickle_file)

            # create a dictionary with the splits and the corresponding image paths
            return {split: train_test_val_splits[i] for i, split in enumerate(splits)}

        # catch errors that occur while unpickling the file
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Error unpickling file {path_pickle_file}: {e}") from e

        # any other unexpected error
        except Exception as e:
            raise Exception(
                f"An unexpected error occurred while loading the pickle file: {e}"
            ) from e

    def _load_split_overview_from_txt(self, version: str) -> dict:
        """Loads the image path in the corresponing splits from diffrent .txt files into a dictionary

        Loads the image paths for the training, test, and validation splits from the .txt files based
        on the prepared data path inside the split_file_paths dictionary with the version string as key.


        Args:
           version (str): Version of the ZooLake dataset as string to load the splits from split_file_paths dictionary

        Returns:
            dict : A dictionary containing the imagepath as value and the corresponding split as key.
                    Example:
                    {
                        "train": [path/to/image1, path/to/image2, ...],
                        "test": [path/to/image3, path/to/image4, ...],
                        "val": [path/to/image5, path/to/image6, ...]
                    }
        """
        splits = ["train", "test", "val"]

        # get the paths from the prepared paths dictionary
        path_txt_files = self._split_file_paths[version]

        # create a dictionary with the splits and the corresponding image paths
        return {split: np.loadtxt(path_txt_files[split], dtype=str) for split in splits}

    def _add_split_column(self, df, image_paths, split_name):
        """Add a column to the DataFrame indicating whether an image is in the split of the given column name

        Args:
            df (pd.DataFrame): DataFrame containing the image names
            image_paths (list): List of image paths for the split
            split_name (str): Name of the split column to add

        Returns:
            pd.DataFrame: DataFrame containing a new column indicating the split.
        """

        # extract the image names from the image paths
        lst = [os.path.basename(image_path) for image_path in image_paths]

        # add a column to the DataFrame indicating whether the image is in the split or not
        df[split_name] = df["image"].isin(lst)

        return df

    def _apply_split_columns_to_dataframe(self, images_paths_split, version, df):
        """Applies __add_split_column to the DataFrame for each split in the images_paths_split dictionary

        Helper function to add the split columns to the DataFrame for each split in the images_paths_split
        dictionary. The column name is generated based on the split name and the version of the ZooLake dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the image metadata and hashes
            images_paths_split (dict): Dictionary containing the image paths for the training, test, and validation splits
            version (str): Version of the ZooLake dataset

        Returns:
            df: DataFrame containing the training set and test set as columns
        """

        for split_name, image_paths in images_paths_split.items():

            # generate the column name based on the split name and the version
            column_name = f"{split_name}_v{version}"

            df = self._add_split_column(
                df=df, image_paths=image_paths, split_name=column_name
            )

        return df
    
    def apply_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the corresponding helper function for each different prepraed split path

        Iterates over the different prepared paths for each version of the ZooLake dataset, orchestrating
        the loading of the images names for each version split. After loading the image paths the data frame
        is updated by applieng a helper function that creates a columns indicating the train/test/val
        split for each version.


        - For version 1, a own helper class for handling .txt split files is used to load the image names.
        - For other versions, the helper function for loading the pickle file is used.

        Args:
            df : DataFrame containing at least the image file names

        Returns:
            The input DataFrame with additional columns indicating the correspondity to the training, test, or validation per version
        """
        # Loop through the different versions of the ZooLake dataset
        for version in self._split_file_paths.keys():

            if version == "1":
                # if the version is 1, load the split and corresponding name from the txt files
                images_paths_split = self._load_split_overview_from_txt(version)

            else:
                # load the split from the pickle file
                images_paths_split = self._load_split_overview_from_pickle(version)

            self.image_splits.update({version: images_paths_split})

            # update the DataFrame with the splits
            df = self._apply_split_columns_to_dataframe(images_paths_split, version, df)

        return df