import itertools
import logging
import os
import warnings
import pandas as pd

from lit_ecology_classifier.checks.duplicates import check_duplicates
from lit_ecology_classifier.data_overview.utils.image_processing import ProcessImage
from lit_ecology_classifier.data_overview.utils.raw_split_preparer import RawSplitPathPreparer
from lit_ecology_classifier.data_overview.utils.raw_split_applier import RawSplitApplier


class OverviewCreator:
    """Create an overview of the images in the difffrent ZooLake dataset versions

    The OverviewCreator is used to create an overview of the images based on the diffrent 
    ZooLake dataset versions that are provided.
    """
    raw_split_applier = RawSplitApplier
    raw_split_preparer = RawSplitPathPreparer

    def __init__(self, zoolake_version_paths: dict = None, hash_algorithm: str = None):

        # Initialize the hash algorithm and check for validity
        self.hash_algorithm = self._init_hash_algorithm(hash_algorithm)

        # Initialize the ZooLake dataset versions with the given paths or default paths and check for existence
        self.zoolake_version_paths = self._init_zoolake_version_paths(
            zoolake_version_paths
        )

        # Prepare the image paths for all images in the dataset versions
        self.image_paths = self._prepare_image_paths()

        # Initialize the image processor
        self.image_processor = ProcessImage(hash_algorithm=self.hash_algorithm)
        self.split_applier = None  #Typ: Optional[SplitApplier]
        self._images_list = []
        self._overview_df = None
        self._overview_with_splits_df = None
        self._duplicates_df = None

    def _init_zoolake_version_paths(self, zoolake_version_paths: dict) -> dict:
        """Initialize the ZooLake dataset versions with the given paths or default paths and check for existence

        Args:
            zoolake_version_paths (dict): Dictionary containing the paths to the different ZooLake dataset versions

        Returns:
            dict: Dictionary containing the paths to the different ZooLake dataset versions
        """
        # If no paths are provided, use the default paths
        if zoolake_version_paths is None:
            zoolake_version_paths = {
                "1": os.path.join("data", "raw", "data"),
                "2": os.path.join("data", "raw", "ZooLake2"),
            }

        # check if each path exists
        for path in zoolake_version_paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path} does not exist")

        return zoolake_version_paths

    def _init_hash_algorithm(self, hash_algorithm: str) -> str:
        """Initialize the hash algorithm with the given value or default to sha256 and check for validity

        Args:
            hash_algorithm (str): Hash algorithm to use for hashing images

        Returns:
            str: Hash algorithm to use for hashing images
        """
        # if no hash algorithm is provided, default to sha256
        if hash_algorithm is None:
            hash_algorithm = "sha256"

        if hash_algorithm not in ["sha256", "phash"]:
            raise ValueError(
                f"Invalid hash algorithm: {hash_algorithm}. Please use 'sha256' or 'phash'"
            )

        return hash_algorithm

    def _collect_image_paths_from_folder(self, version_path) -> list[str]:
        """Prepares a list of file paths for all images with a `.jpeg` extension in the given folder recursively.

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
        """  Prepares a list of dictionaries containing the dataset version and the image paths

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
                self.zoolake_version_paths.items(),
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

        version_image_pairs = itertools.chain.from_iterable(
            map(
                lambda item: zip(itertools.repeat(item[0]), item[1]),
                self.image_paths.items(),
            )
        )



        # Process all images using map
        processed_images = map(
            lambda version_image: self.image_processor.process_image(*version_image),
            version_image_pairs,
        )

        # extend the list with the dictionary of the processed images
        self._images_list.extend(processed_images)

        return self._images_list

    def _ood_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixes the 'OOD' values in the 'data_set_version' column

        The OOD are originaly part of the ZooLake version 2 but are provided in a different folder/zip
        With the current implementation, the 'OOD' would be a own dataset version. To fix this, the 'OOD'
        values are replaced with '2' in the 'data_set_version' column. A new column 'OOD_v2' is added
        to indicate the original 'OOD' values.

        Args:
            df (pd.DataFrame): A DataFrame containing the image metadata and data_Set_versions
                                Example:
                                | image | class | data_set_version |
                                |-------|-------|------------------|
                                | img2  | class2| 2                |

        Returns:
            pd.DataFrame: Df containing the image metadata and hashes with 'OOD' values fixed.
                            Example:
                            | image | class | data_set_version | OOD_v2 |
                            |-------|-------|------------------|--------|
                            | img1  | class1| 2                | True   |
                            | img2  | class2| 2                | False  |


        """
        if df["data_set_version"].str.contains("OOD").any():
            warnings.warn(
                "Found 'OOD' in 'data_set_version' column. Assuming Version 2"
            )

            # Add a column to indicate where 'OOD' was found
            df["OOD_v2"] = df["data_set_version"].str.contains("OOD")

            # Replace 'OOD' with '2' in 'data_set_version' values
            df["data_set_version"] = df["data_set_version"].str.replace(
                "OOD", "2", regex=False
            )

        return df

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
            df, columns=["data_set_version"], drop_first=False, prefix="version"
        )

        # group by hash value an keep the maximum value of each columns in the group and preserve the columns order
        df = df.groupby(
            ["image", "class", self.hash_algorithm], as_index=False, sort=False
        ).max()

        return df

    def _add_split_columns(self, df: pd.DataFrame, split_paths: dict) -> pd.DataFrame:
        split_applier = self.raw_split_applier(split_paths)
        df = split_applier.apply_splits(df)
        return df

    def get_duplicates_df(self):
        """Get duplicates dataframe"""
        if self._duplicates_df is None:
            df = self.get_raw_df()
            self._duplicates_df = check_duplicates(df, by_data_set_version=True)
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
            df = self._ood_fixes(df)
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

        split_paths = self.raw_split_preparer(
            self.zoolake_version_paths
        ).prepare_split_paths()

        if not self._images_list or load_new:
            self.get_raw_df()

        if self._overview_df is None or load_new:
            self.get_overview_df()

        if self._overview_with_splits_df is None or load_new:
            self._overview_with_splits_df = self._add_split_columns(
                self._overview_df, split_paths=split_paths
            )

        return self._overview_with_splits_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    overview_creator = OverviewCreator()
    overview_df = overview_creator.get_overview_with_splits_df()
    print(overview_df.head())
