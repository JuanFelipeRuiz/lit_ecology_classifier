"""
Main module for the split process. Provides all necessary functions to check for used splits
and the creation of new splits based on the provided split strategy, data set version and out
The moving of the images is handled by the SplitImageMover class.
"""

import importlib
import logging
import os


import pandas as pd
from ..splitting.split_strategies.split_strategy import SplitStrategy
from ..data.mover.split_images_mover import SplitImageMover
from ..helpers.hashing import HashGenerator


logger = logging.getLogger(__name__)


class SplitProcessor:
    """
    Provides the main functionalities to manage the split process. It includes the
    checking of existing splits, the creation of new splits aand executing the image
    moving based on the split. To ensure the correct reproducibility of the split and
    to simplify the checking of existing splits, the used images hashes are reduced to
    a single hash value.

    Attributes:
        split_strategy (SplitStrategy): Split strategy to be used.
        dataset_version (str): Version of the dataset,
        ood_version (str): Out-of-Distribution (OOD) version.
        image_overview_path (str): Path to the image overview CSV.
        split_overview_path (str): Path to the split overview CSV.
        image_base_paths (str): Path to the image directory.

    Inheritance:
        SplitProcessor (SplitProcessor): Base class for methods

    """

    def __init__(
        self,
        split_strategy = None,
        dataset_version = None,
        ood_version=None,
        image_overview_path=None,
        split_overview=None,
        image_base_paths=None,
        tgt_base_path=None,
    ):
        """Initializes the SplitProcessor with necessary parameters.

        Args:
            split_strategy: An instance of a SplitStrategy.
            dataset_version (str): Version of the dataset.
            ood_version (str, optional): Out-of-Distribution (OOD) version.
            image_overview_path (str, df): Df containing the image overview or path to CSV.
            split_overview_path (str, df): Df containing the split overview or path to CSV.
            image_base_paths (str, optional): Bas epath to the image directory.
        """
        self.imgage_mover = self._init_image_mover(image_base_paths, tgt_base_path)
        self.image_overview_df = self._init_image_overview_df(image_overview_path)
        self.split_overview_df = self._init_split_overview_df(split_overview)
        self.ood_version = ood_version
        self.dataset_version = dataset_version
        self.split_strategy_str, self.split_strategy_instance  = self._init_split_strategy(split_strategy)
        self.split_df = None

    def _init_split_strategy(self, split_strategy):
        """Initializes the split strategy 

        Checks if the provided split strategy is a valid class or just a string. If it is a string,
        no class is provided

        Args:
            split_strategy (str): Name of the split strategy class.

        Returns:
            SplitStrategy_str: name of the split strategy class.
            SplitStrategy_class: Instance of the split strategy class.
        """

        if isinstance(split_strategy, str):
            split_strategy_str = split_strategy
            return split_strategy_str, None
            
        elif isinstance(split_strategy, SplitStrategy):
            split_strategy_str = split_strategy.__class__.__name__
            split_strategy_instance = split_strategy
            return split_strategy_str, split_strategy_instance

        else:
            raise ValueError("split_strategy must be a string or an instance of SplitStrategy.")
        
    def _search_split_strategy(self):
        """Search for the split strategy inside the split_strategies folder.
        """
        # try to import the split strategy module
        try:
            modul = importlib.import_module(
                "lit_ecology_classifier.splitting.split_strategies." + self.split_strategy_str
            )
        except ModuleNotFoundError as e:
            logger.error("Split strategy %s not found in registry.", self.split_strategy_str)
            raise e
        

        try:
            imported_class = getattr(modul, self.split_strategy_str.capitalize())
            self.split_strategy_instance = imported_class()

        except AttributeError as e:
            logger.error("Split strategy %s not found inside the module %s", 
                         self.split_strategy_str,
                         modul)
            raise e
        return self.split_strategy_instance

    def _init_image_overview_df(self, overview_df: pd.DataFrame = None) -> pd.DataFrame:
        """Initializes the image overview DataFrame.

        If no DataFrame is provided, a default CSV is loaded.

        Args:
            overview_df (pd.DataFrame|str ): Df containing the columns image names and class labels.  
                                             Or path to the image overview CSV.  

        Returns:
            pd.DataFrame: Loaded image overview DataFrame.
        """

        # if no overview_df is provided, load the default overview.csv
        if overview_df is None:
            default_path = os.path.join("data", "interim", "overview.csv")
            types = {
                "dataset_version": "str",
                "OOD": "str",
                "split_strategy": "str",
                "combined_split_hash": "str",
                "description": "str"
            }
            return pd.read_csv(default_path, dtype=types)
        if type(overview_df) == str:
            return pd.read_csv(overview_df)
        
        return overview_df

    def _init_image_mover(
        self, image_base_paths: None | str, tgt_path: None | str
    ) -> SplitImageMover:
        """Initializes the image paths. If not provided, a default basepath is used.

        Args:
            image_base_paths (str): Path to the image directory. Default path: "data/interim/ZooLake".
            tgt_path (str): Path to the target directory. Default path: "data/processed".

        Returns:
            SplitImageMover: Instance of the SplitImageMover class.

        """
        if image_base_paths is None:
            image_base_paths = os.path.join("data", "interim", "ZooLake")
            if not os.path.exists(image_base_paths):
                logger.error("Image base path not found:%s", image_base_paths)
                raise FileNotFoundError(
                    f"Image base path not found: {image_base_paths}, please provide a valid path."
                )

        if tgt_path is None:
            tgt_path = os.path.join("data", "processed")
            if not os.path.exists(tgt_path):
                logger.error("Target path not found:%s", tgt_path)
                raise FileNotFoundError(
                    f"Target path not found: {tgt_path}, please provide a valid path."
                )

        return SplitImageMover(src_base_path=image_base_paths, tgt_base_path=tgt_path)

    def _init_split_overview_df(
        self,
        split_overview: None | pd.DataFrame,
    ) -> pd.DataFrame:
        """Initializes the split overview DataFrame.

        If no Split overview is provided, a default CSV is loaded or a new one is created.

        Args:
            split_overview_path (pd.DataFrame, optional): DataFrame containing split overview.

        Returns:
            pd.DataFrame: Loaded or newly created split overview DataFrame.
        """
        if split_overview is None:
            default_path = os.path.join(
                "data", "interim", "train_test_val", "split_overview.csv"
            )
            if not os.path.exists(default_path):
                logger.warning(
                    "General split overview not found. Creating a new one at:%s",
                    {default_path},
                )
                type_dict = {
                    
                    "dataset_version" : "str",
                    "split_strategy":"str",
                    "ood_version":"str",
                    "combined_split_hash":"str",
                    "description" : "str",
                    
                }
            return pd.read_csv(default_path, dtype=type_dict)
        return split_overview

    def _prepare_filepath(self, hash_value: str) -> str:
        """Prepares the file path based on the provided hash.

        Args:
            hash_value (str): Hash value.

        Returns:
            str: Prepared file path.
        """
        filename = f"{str(hash_value)[:24]}.csv"
        filepath = os.path.join("data", "interim", "UsedSplits", filename)
        return filepath

    def _reload_split(self, df):
        """Reloads an existing split based on the hash.

        Args:
            existing_split_df (pd.DataFrame): DataFrame containing existing split information.

        Returns:
            pd.DataFrame: Reloaded split DataFrame. Example:

                    |image|class|split|
                    |-----|-----|-----|
                    |img1 |1    |train|
                    |img2 |2    |val  |
                    |img3 |3    |test |
        """
        logger.info("Existing split found. Reloading split.")
        hash_value = df["combined_split_hash"].values[0]
        filepath = self._prepare_filepath(hash_value)
        if not os.path.exists(filepath):
            logger.error("Split file not found:%s", filepath)
            raise FileNotFoundError(f"Split file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def _get_version_col_name(self, df):
        """Prepare the version column based on the dataset and ood version.

        Args:
            df (Dataframe): Dataframe containing the image names and the class labels for the split

        Returns: Dataframe containing the image names and the class labels for the split
        """

        version_col_name = "version_" + self.dataset_version
        
        if version_col_name not in df.columns:
            logger.error("Dataset version not found in image overview")
            raise ValueError("Dataset version not found in image overview")
            
        
        return version_col_name


    def _filter_df(self, df):
        """Filter the overview dataframe based on the dataset and out of dataset (ood) version.

        To prevent the split of images that are not part of the dataset version or are part of
        the ood version. It ensures the repoducibility of old models by providing the same data
        version.

        Args:
            df (Dataframe): Dataframe containing the image names and the class labels for the split


        Returns: Filtered Dataframe containing the image names and the class labels for the split
        """
        logger.debug("Filtering data based on dataset version: %s", self.dataset_version)
   
        version_col_name = self._get_version_col_name(df)

        df = df[df[version_col_name] == True]

        logger.debug("Filtering dataframe: %s", df.shape)
        if self.ood_version is not None:
            df = df[df["ood_version"] != self.ood_version]

        return df

    def execute_split(self, df: pd.DataFrame, **kwargs) -> dict:
        """
        Executes the split using the provided split strategy.

        Args:
            df (pd.DataFrame): Filtered DataFrame.
            **kwargs: Additional parameters for the split strategy.

        Returns:
            dict: Dictionary containing split data.
        """

        if self.split_strategy_instance is None:
            self.split_strategy_instance = self._search_split_strategy()
            
        return self.split_strategy_instance.perform_split(df, **kwargs)

    def _transform_split_dict(self, split_dict: dict) -> pd.DataFrame:
        """Transform the split dictionary into a dataframe with the columns image and split.

        Args:
            split_dict (dict): Dictionary containing the split data. Example:
                {
                    "train": img1, img2, ...
                    "val": img3, img4, ...
                    "test": img5, img6, ...
                }

        Returns:
            pd.DataFrame: DataFrame containing the split data and further informations. Example:
                        |image|split|
                        | ---- | ---- |
                        |img1 |train|
                        |img2 |train|
        """

        # List to store DataFrames with split labels
        combined_list = []

        # Iterate over each split (key) and its corresponding data (value) in the dictionary
        for split_name, (X, y) in split_dict.items():
    
            split_df = pd.concat([X, y], axis=1)
            # Add the "split" column with the split name (e.g., "train", "test", etc.)
            split_df["split"] = split_name
            # Append to the list of DataFrames
            combined_list.append(split_df)

        # Concatenate all split DataFrames into one DataFrame
        combined_df = pd.concat(combined_list, ignore_index=True)

        logger.debug("Split data transformed successfully.")
        return combined_df


    def _merge_split_df(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Merge the split dataframe with the image overview dataframe.

        Args:
            split_df (Dataframe): Dataframe containing the image names and split labels. 
                                 Example:
                                    |image|split|
                                    |-----|-----|
                                    |img1 |train|
                                    |img2 |val  |


        Returns:
            pd.DataFrame: Merged DataFrame. Example:
                            |image|class|hash|split|
                            |-----|-----|----|-----|    
                            |img1 |1    |hash|train|
                            |img2 |2    |hash|vall|


        """

        df_to_merge = self.image_overview_df[["image", "class", "sha256"]]
        return split_df.merge(
            df_to_merge,  on=["image","class"], how="left"
        )
       

    def append_split_overview(self, 
                              combined_split_hash: str,
                              description: str = None):
        """
        Appends the split metadata to the split overview DataFrame.

        Args:
            split_df (pd.DataFrame): Split DataFrame.
            combined_split_hash (str): Combined split hash.
        """
        new_entry = {
            "dataset_version": self.dataset_version,
            "split_strategy": self.split_strategy_str,
            "ood": self.ood_version,
            "combined_split_hash": combined_split_hash,
            "description": description,
        }

        row_to_append = pd.DataFrame(new_entry, index=[0])
        self.split_overview_df = pd.concat([self.split_overview_df, row_to_append], ignore_index=True)
        logger.info("Split overview updated.")

    def save_split(self, split_df: pd.DataFrame, combined_split_hash: str):
        """
        Saves the split DataFrame to a CSV file.

        Args:
            split_df (pd.DataFrame): Split DataFrame.
            combined_split_hash (str): Combined split hash.
        """
        filepath = self._prepare_filepath(combined_split_hash)
        split_df.to_csv(filepath, index=False)
        logger.info("Split saved at: %s", filepath)

    def save_split_overview(self):
        """Saves the split overview DataFrame to a CSV file.

        Args:
            split_df (pd.DataFrame): Split DataFrame.
            combined_split_hash (str): Combined split hash.
        """
        self.split_overview_df.to_csv(
            os.path.join("data", "interim", "USedSplits", "split_overview.csv"),
            index=False,
        )
        logger.info("Split overview saved.")


    def _create_split(self) -> pd.DataFrame:
        """Create a new split based on the provided DataFrame.

        Mangaes the entire split process by filtering the DataFrame, executing the split,
        transforming the split data and generating hashes for the split data.
        """
        filtered_df = self._filter_df(self.image_overview_df)
        split_dict = self.execute_split(filtered_df)
        logger.info("Splitted data successfully with %s", self.split_strategy_str)
        logger.debug("Starting transformation of split%s", split_dict)
        split_df = self._transform_split_dict(split_dict)
        logger.debug("Starting merge of split with overview df.")
        split_df = self._merge_split_df(split_df)
        logger.debug("Starting generation of hashes.")
        split_hashes = HashGenerator.generate_hash_dict_from_split(split_df, col_to_hash="sha256")
        combined_split_hash = HashGenerator.sha256_from_list(split_hashes.values())

        self.append_split_overview(combined_split_hash)


        return split_df
    
    def save_changes(self):
        """Saves the changes to the split overview."""  
        self.save_split(self.split_df, self.split_overview_df["combined_split_hash"].values[0])
        self.save_split_overview(self.split_overview_df, self.split_overview_df["combined_split_hash"].values[0])

    def copy_images(self):
        """Copies the images based on the split DataFrame."""
        self.imgage_mover.copy_images(self.split_df)
        logger.info("Images copied successfully.")


    def main(self) -> pd.DataFrame:
        """
        Executes the entire split process.

        Returns:
            pd.DataFrame: Final split DataFrame.
        """
        existing_split = self.split_overview_df[
            (self.split_overview_df["dataset_version"] == self.dataset_version)
            & (
                self.split_overview_df["split_strategy"]
                == self.split_strategy_str
            )
        ]

        if not existing_split.empty:
            logger.info("Existing split found, reloading split.")
            split_df = self._reload_split(existing_split)
        else:
            logger.info("No existing split found, creating new split.")
            split_df = self._create_split()

        self.split_df = split_df
        return split_df

