from datetime import datetime as dt
from datetime import timezone
import itertools
import hashlib
import pickle
import os
import warnings

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

class CreateOverviewDf:
    """Generate an overview DataFrame based on the different ZooLake dataset versions, including images and metadata.

    Image Processing and Metadata Extraction:
        Each image in the dataset is processed individually to extract metadata from its file name and folder path.
        The SHA256 hashing algorithm is used to calculate a unique hash value for each image, which enables identical
        duplicate detection by identifying images with the same hash value. The hash value also facilitates merging 
        images from different dataset versions into a single DataFrame, with a one-hot encoded column indicating in
        which data set versions the image occurs.

    Additional columns can be generated to indicate in which split (train, test, or validation) the image appears,
    based on a pickle files. For version 1, the split information is stored in separate .txt files and is also 
    implemented in this class. The process assumes that the images are stored externally with the original
    structure of the corresponding ZooLake dataset version.

    Attributes:
        zoolake_version_paths (dict): Maps dataset versions to their corresponding file paths.

        split_file_paths  (dict): Maps dataset versions to their corresponding file paths for the 
                                    train/test/validation splits.

        hash_algorithm (str): Specifies the hashing algorithm to use, either "sha256" or "phash".

        _images_list (list): List of dictionaries containing the image metadata and hashes for all images
                                in the given dataset versions

        _overview_df (pd.DataFrame): DataFrame containing image metadata and hash values.

        overview_with_splits_df (pd.DataFrame): DataFrame containing image metadata, hashvalue and columns indicating which split 
                                                    (train/test/validation) the image belongs to.

        duplicates_df (pd.DataFrame):  DataFrame listing duplicate images in the dataset based on hash values.

    Main methods:
        main: Main function to create the overview DataFrame with splits.
        get_raw_df: Get the raw DataFrame containing only the the image metadata and hashes, without any further processing.
        get_overview_df: Get the overview DataFrame grouped by hashvalue, data set version hot encoded and without train/test/val splits
        get_overview_with_splits_df: Get the overview DataFrame with the train/test/val splits included, alternative to main method.
        get_duplicates_df: Get the DataFrame containing the duplicates in the dataset grouped on hash values and data set version.
    """

    def __init__(self, zoolake_version_paths: dict = None, hash_algorithm: str = None):
        """Initialize the overview DataFrame creator based on the dataset versions and hashing algorithm

        Args:
            zoolake_version_paths: A dictionary that maps the dataset versions to their corresponding file paths.
            hash_algorithm : String that specifies the hashing algorithm to use, either "sha256" or "phash".

        """

        if zoolake_version_paths is None:
            zoolake_version_paths = {
                "1": os.path.join("data", "raw", "data"),
                "2": os.path.join("data", "raw", "ZooLake2"),
            }

        self.zoolake_version_paths = zoolake_version_paths
        self.__zoolake_version_paths()

        if hash_algorithm is None:
            hash_algorithm = "sha256"

        if hash_algorithm not in ["sha256", "phash"]:
            raise ValueError(
                f'Invalid hash algorithm: {hash_algorithm}. Choose between "sha256" and "phash"'
            )

        self.image_paths = self._prepare_images_paths()

        self.hash_algorithm = hash_algorithm

        self._split_file_paths = None
        self._images_list = []

        self._overview_df = None
        self._overview_with_splits_df = None
        self._duplicates_df = None

    def __zoolake_version_paths(self) -> None:
        for path in self.zoolake_version_paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path} does not exist")

    def _hash_image_sha256(self, image_path: str) -> str:
        """Calculate the hash from the binary data of the image using the SHA256 algorithm.

        Args:
            image_path (str): Path to the image file as string

        Returns:
            str: Hash value of the image as string

        Raises:
            PermissionError: If the image cannot be read due to permission issues
            Exception: If the image cannot be hashed due to other issues
        """
        try:

            with Image.open(image_path) as img:

                #  if image_path ends with .jpeg or .png
                if img.format not in ["JPEG", "PNG"]:
                    raise ValueError(
                        f"Error hashing {image_path} with SHA256: Invalid file format {img.format}"
                    )

                # Read the binary data of the image
                img_data = img.tobytes()

                # Calculate the SHA256 hash based on the binary data of the image
                return hashlib.sha256(img_data).hexdigest()

        except PermissionError as pe:
            raise PermissionError(f"Permission denied when accessing {image_path}:{pe}") from pe

        except Exception as e:
            raise Exception(f"Error hashing {image_path} with SHA256: {e}") from e

    def _hash_image_phash(self, image_path: str) -> str:
        """Calculate the perceptual hash (pHash) from the binary data of the image.

        Perceptual hashing is useful for detecting visually similar images. However, in the usage with the ZooLake Dataset,
        where all images have a black background and the plankton species are nearly identical to each other,
        pHash tends to generate a high number of false positives. As a result, it is not recommended to use pHash for detecting similar
        duplicates.

        Args:
            image_path (str): Path to the image file as string

        Returns:
            str: pHash value of the image

        Raises:
            PermissionError: If the image cannot be read due to permission issues
            Exception: If the image cannot be hashed due to other issues
        """
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))

        except PermissionError as pe:
            raise PermissionError(f"Permission denied when accessing {image_path}: {pe}") from pe

        except Exception as e:
            raise Exception(f"Error hashing {image_path} with phash: {e}") from e

    def _extract_timestamp_from_filename(self, image_path: str) -> dt:
        """Extract the timestamp from the image filename and convert it to a datetime object in UTC timezone.

        The timestamp (without miliseconds) is expected to be at a fixed position in the filename (characters 15-25).
        This function will extract those characters, convert them to an integer timestamp, and return a UTC datetime object.

        Args:
            image_path (str): Path to the image file as string to extract the timestamp from

        Returns:
           dt: Timestamp extracted from the filename as a datetime object with UTC as timezone

        Raises:
            ValueError: If the extracted value cannot be converted to a timestamp
        """

        try:

            # Extract the image name from the path
            image_name = os.path.basename(image_path)

            # Extract the timestamp part and keep only the first 10 characters (ignoring mili seconds)
            timestamp_str = image_name[15:25]

            # return the timestamp as a datetime object with UTC as timezone
            return dt.fromtimestamp(int(timestamp_str), tz=timezone.utc)

        except IndexError as ie:

            raise ValueError(
                f"Error extracting timestamp: Failed slicing timestamp from '{image_name}:{ie}'"
            ) from ie

        except Exception as e:
            raise ValueError(
                f"Error extracting and creating timestamp from {image_path}: {e}" 
            ) from e

    def _extract_plankton_class(self, image_path: str, version: str) -> str:
        """Extract  plankton class from the image path based on version.

        The extraction method is based on the different dataset versions:
        - For version 1, the plankton class is extracted from the grandparent directory,
            since there is an additional 'training' folder .
        - For other versions, the plankton class is extracted from the immediate parent directory.

        Args:
            image_path (str): Path to the image file
            version (str): Version of the ZooLake dataset as string

        Returns:
            str: The plankton class name
        """
        try:
            if version == "1":

                # Get the second parent / grandparent directory for ZooLake version 1
                return os.path.basename(os.path.dirname(os.path.dirname(image_path)))

            # get the parent directory for each other ZooLake version
            return os.path.basename(os.path.basename(os.path.dirname(image_path)))

        except Exception as e:
            raise ValueError(
                f"Error extracting plankton class from {image_path}: {e}"
            ) from e

    def process_image(self, version, image_path) -> dict:
        """Process a single image to calculate the hash and extract metadata from the filename and path.

        Args:
            version_path_tuple (tuple) : tuple containing
                image_path (str): Path to the image file
                version (str): Version of the ZooLake dataset as string for metadata

        Returns:
            dict: Dictionary containing the image metadata and hashes. Example:
                {
                    "image": "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.",
                    "sha256": "a957e3fb302aa924ea62f25b436893151640dc05f761c402d3ec98782b801b34e",
                    "class": "aphanizomenon",
                    "data_set_version": "1",
                    "date": "2019-10-08 14:02:52+00:00"
                }

        Raises:
            Exception: If the image cannot be processed
        """
        try:

            image_date = self._extract_timestamp_from_filename(image_path)
            plankton_class = self._extract_plankton_class(image_path, version)

            image_metadata = {
                "image": os.path.basename(image_path),
            }

            if self.hash_algorithm == "phash":
                image_phash = self._hash_image_phash(image_path)
                image_metadata.update({"phash": image_phash})

            elif self.hash_algorithm == "sha256":
                image_hash_sha256 = self._hash_image_sha256(image_path)

                image_metadata.update({"sha256": image_hash_sha256})

            image_metadata.update(
                {
                    "class": plankton_class,
                    "data_set_version": version,
                    "date": image_date,
                }
            )

            return image_metadata

        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {e}") from e

    def _collect_image_paths_from_folder(self, version_path) -> list[str]:
        """Prepares a list of file paths for all images with a `.jpeg` extension in the given folder recursively.

        Searches the specified folder and its subfolders recursively for image files with a `.jpeg` extension.
        It returns the full paths to each image found,  preparing the list of image paths for further processing.

        Args:
            version_path) (tuple): Conaints version and path to the class. Example:
                            ("1": "path/to/zoolakev1")

        Returns:
            list[str]: List of found image paths
        """
        version, folder_path = version_path

        image_path = [
            # join the root path with the file name
            (version, os.path.join(root, file))
            # walk through the folder and subfoledrs (generates lists of filespath and filenames)
            for root, _, files in os.walk(folder_path)
            # loop through the files in the folder
            for file in files
            # filter for files that end with .jpeg
            if file.endswith(".jpeg")
        ]

        return image_path

    def _prepare_images_paths(self) -> list[dict]:
        """Generate a list of dictionaries containing the image metadata and hashes for all images in the dataset versions.

        Returns:
            self._images_list : List of dictionaries containing the image metadata and hashes for all images in the dataset versions
        """

        return list(
            itertools.chain.from_iterable(
                map(self._collect_image_paths_from_folder,self.zoolake_version_paths.items(),
                )
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

        # add a list with the same lenght of the file_list to give the version
        processed_images = map(self.process_image, *zip(*self.image_paths))

        # extend the list with the dictionary of the processed images
        self._images_list.extend(processed_images)

        return self._images_list

    def check_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for duplicates in the same dataset version

        The duplicates are identified by comparing the hash values of the images in each dataset.
        When a duplicate is found, a DataFrame is created with further information, like if the duplicate
        images have the same class or image name. A duplicate with diffrent classes can have more negative
        impact than a duplicate with the same class, since it can lead to misclassification.

        Args:
            df : DataFrame containing the image metadata and hashes to  for duplicates

        Returns:
            A new DataFrame containing the duplicates in the dataset based on hash values
        """

        if self.hash_algorithm != "sha256":
            warnings.warn(
                "Duplicates  is only available for sha256 hash algorithm since \
                    the phash is not unique for similar images and raises false positives"
            )

        # save the duplicates hash values in a list
        duplicates = df[
                df.duplicated(subset=["sha256", "data_set_version"], keep=False)
            ].copy()

        if duplicates.empty:
                
                print("No duplicates found in the dataset")
                return None
        else: 
                # Group by hash_col and DataSetVersion
            group_counts = (
                duplicates.groupby(["sha256", "data_set_version"])
                .agg(
                    # Count the number of duplicates
                   count=("class", "size"),
                    # Check if the class and image name are the same for all duplicates
                    diffrent_class=("class", lambda x: x.nunique() != 1),
                    diffrent_image_name=("image", lambda x: x.nunique() != 1),
                    )
                .reset_index()
            )

            if group_counts["diffrent_image_name"].all() is False:

                group_counts = pd.merge(
                    group_counts,
                    df[["sha256", "data_set_version", "image", "class"]],
                    on=["sha256", "data_set_version"],
                    how="left",
                )

            group_counts["count"] = group_counts["count"].astype(int)

            self._duplicates_df = group_counts[group_counts["count"] > 0]

            warnings.warn(f"Duplicates found in the dataset: {duplicates.shape[0]}")
            
            return self._duplicates_df
                

       

    def _add_one_hot_encoded_versions_and_group_by(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """add One-hot encode columns for each data set versions in the DataFrame and group by the hash column, image name and class.

        One-hot encodes the `data_set_version` column and groups the DataFrame by the image name, class
        and the hash column (based on the hash algorithm being used). Within each group, the maximum /latest value
        of the timestamp column is retained. This process makes the DataFrame tidy and ready for further analysis.

        Args:
            df (pd.DataFrame): A DataFrame containing the image metadata and hashes

        Returns:
            pd.DataFrame: DataFrame containing the tidy image metadata and hashes with one-hot encoded data set versions columns.

        Raises:
            ValueError: If the required columns are missing in the DataFrame
        """

        required_columns = ["image", "class", "data_set_version", self.hash_algorithm]

        # Check if the required columns are present in the DataFrame to prevent errors
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The following required columns are missing: {missing_columns}"
            )

        # One-hot encode the data_set_versions column
        df = pd.get_dummies(
            df, columns=["data_set_version"], drop_first=False, prefix="ZooLake"
        )

        # group by hash value an keep the maximum value of each columns in the group and preserve the columns order
        df = df.groupby(
            ["image", "class", self.hash_algorithm], as_index=False, sort=False
        ).max()

        return df

    def _prepare_split_paths_from_txt(self, datapath: str, version: str) -> dict:
        """Pepare and  if the txt files containing thesplits exist in the data folder for version 1.

        It assumes that the txt files are stored in a folder called 'zoolake_train_test_val_separated'
        within the provided datapath. This only applies to version 1. After ing the existence of the txt files,
        it generates the paths to the train/test/val .txt files and stores them in a dictionary.

        Args:
            datapath (str): Path to the folder containg the data for ZooLake dataset V1 as string
            version (str): Version of the ZooLake dataset as string to generate a key for the path inside the path dictionary

        Returns:
           dict: Dictionary containing the data set version and subdictionory with the paths to the train/test/val .txt files
                 as values and the split name as key.

        Raises:
            FileNotFoundError: If a txt files do not exist in the folder
        """
        # generate path to the folder containing the txt files
        path_txt_file_folder = os.path.join(
            datapath, "zoolake_train_test_val_separated"
        )

        # create dict to store the paths to the diffrent split .txt files
        self._split_file_paths[version] = {
            "train": os.path.join(path_txt_file_folder, "train_filenames.txt"),
            "test": os.path.join(path_txt_file_folder, "test_filenames.txt"),
            "val": os.path.join(path_txt_file_folder, "val_filenames.txt"),
        }

        #  if a txt file does not exist in the folder
        missing_files = [
            file
            for file in self._split_file_paths[version].values()
            if not os.path.exists(file)
        ]

        # raise a error if a file is missing
        if missing_files:
            raise FileNotFoundError(
                f"The following files are missing for version {version}: {', '.join(missing_files)}"
            )

        return self._split_file_paths

    def _prepare_split_paths_from_pickle(self, datapath: str, version: str) -> dict:
        """Search if a pickle file containing the splits exist in the folder

        Searches if they are any pickle files inside the datapath. If the search does not return a result
        it raises a warning, since it could be possible for a new dataset. When the search resuluts with
        multiple pickle files, it raises a valueerror. The pickle file is expected to contain the
          train/test/val splits for the given ZooLake dataset version.

        Args:
            datapath (str): Path to the folder containg the data for the given ZooLake Version as string
            version (str): Version of the ZooLake dataset as string to generate a key for the path inside the path dictionary

        Returns:
            dict: Dictionary containing the path to the pickle file for the train/test/val splits

        Raises:
            FileNotFoundError: If no pickle file is found in the folder
            ValueError: If multiple pickle files are found in the folder, as only one pickle file is expected
        """
        pickle_files = [
            file
            for file in os.listdir(datapath)
            if file.endswith(".pickle") or file.endswith(".pkl")
        ]

        #  if a pickle file exists
        if len(pickle_files) == 0:
            warnings.warn(f"No pickle file for {version}found in {datapath}")
            return None

        #  if there are multiple pickle files
        elif len(pickle_files) > 1:
            raise ValueError(
                f"Multiple pickle files found in {datapath}. Please provide only one pickle file"
            )

        else:

            # Prepare file path for the pickle file
            path_pickle_file = os.path.join(datapath, pickle_files[0])
            self._split_file_paths[version] = {"pickle": path_pickle_file}

        return self._split_file_paths

    def _prepare_split_paths(self) -> dict:
        """Prepare the file paths for the train/test/val splits for the different ZooLake dataset versions

        Prepares the file paths for different versions of the ZooLake dataset.
        - For version 1, it looks for train/test/val splits stored as .txt files.
        - For version 2 (or higher), it assumes that the splits are stored in a pickle file.
        - For version "Q", it skips the preparation, as it is the collection of unlabelled or unreleased images.

        The preparation is unboudled from the loading of the splits, to be able to  the existence of
        missing files at an early stage.

        Args:
            None

        Returns:
            dict: Dictionary containing the paths to the train/test/val txt files or the pickle file

        Raises:
            Warning: If the version is not 2, since currently only version 2 is available and it is assumed
                     that the split is stored in a pickle file
            Warning: If there is no pickle file inside for a version
        """

        self._split_file_paths = {}

        for version, datapath in self.zoolake_version_paths.items():
            if version == "1":
                # Prepare file paths for the txt files (train/test/val)
                self._prepare_split_paths_from_txt(datapath, version)

            elif version == "Q":
                continue

            else:
                # raise warning if the version is not 2, since currently only version 2 is released
                if version != "2":
                    warnings.warn(
                        "New version, assuming a pickle file for split in the folder"
                    )

                self._prepare_split_paths_from_pickle(datapath, version)

        return self._split_file_paths

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

    def _apply_splits_to_dataframe(self, images_paths_split, version, df):
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

    def _process_versions_splits_by_version(self, df: pd.DataFrame) -> pd.DataFrame:
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

            # update the DataFrame with the splits
            df = self._apply_splits_to_dataframe(images_paths_split, version, df)

        return df

    def get_duplicates_df(self):
        """Get duplicates dataframe"""
        if self._duplicates_df is None:
            df = self.get_raw_df()
            self.check_duplicates(df)
        return self._duplicates_df

    def get_raw_df(self):
        """Get the raw DataFrame containing only the image metadata and hashes, without any further processing."""
        #  if the images list is empty
        if not self._images_list:
            self._process_images_by_version()
        return pd.DataFrame(self._images_list)

    def get_overview_df(self):
        """Get the overview DataFrame grouped and the data set version as hot encoded columns."""
        if self._overview_df is None:
            df = self.get_raw_df()
            self.check_duplicates(df)
            self._overview_df = self._add_one_hot_encoded_versions_and_group_by(df)
        return self._overview_df

    def get_overview_with_splits_df(self):
        """Get the overview DataFrame with the train/test/val splits included, alternative to main method."""
        if self._overview_with_splits_df is None:
            self._overview_with_splits_df = self.main()
        return self._overview_with_splits_df


    def main(self, load_new=False):
        """Main function to create the overview DataFrame with columns indicating the belonging to the train/test/val splits.

        Args:
            load_new (bool): If True, the function will reload the data and prepare the splits again.

        Returns:
            pd.DataFrame: DataFrame containing the image metadata and hashes
        """

        if self._split_file_paths is None or load_new:
            self._prepare_split_paths()

        if not self._images_list  or load_new:
            self.get_raw_df()

        if self._overview_df is None or load_new:
            self.get_overview_df()

        if self._overview_with_splits_df is None or load_new:
            self._overview_with_splits_df = self._process_versions_splits_by_version(
                self._overview_df
            )

        return self._overview_with_splits_df


if __name__ == "__main__":
    print("Running the dataset creator")
    dataset_creator = CreateOverviewDf(
        zoolake_version_paths={
            "1": os.path.join("data", "raw", "data"),
            "2": os.path.join("data", "raw", "ZooLake2"),
        }
    )

    # start time measurement
    from time import time

    start = time()
    dataset_creator.main()
    print(f"Time taken: {time() - start}")
    dataset_creator()
