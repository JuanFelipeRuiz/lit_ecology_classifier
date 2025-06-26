import os
import pathlib
from pathlib import Path
import pandas as pd
import logging
from typing import Optional, Union

from lit_ecology_classifier.splitting.filtering.filter_resolver import apply_filter_by_name
from lit_ecology_classifier.splitting.split_strategies.split_resolver import perform_splitting
from lit_ecology_classifier.splitting import split_search
from lit_ecology_classifier.splitting.split_reload import reload_logic
from lit_ecology_classifier.data_overview.utils.base_overview_creator import BaseOverviewCreator
from lit_ecology_classifier.helpers.hashing import HashGenerator
from lit_ecology_classifier.helpers import filter as filter_helpers



logger = logging.getLogger(__name__)


class BaseSplitProcessor:
    def __init__(
        self,
        split_overview: Optional[Union[str, pathlib.Path, pd.DataFrame]] = "split_overview.csv",
        artefacts_folder: Optional[Union[str, pathlib.Path]] = None,
        image_overview: Optional[Union[str, pathlib.Path, pd.DataFrame, BaseOverviewCreator]] = None,
    ):
        self.split_args = {}
        self.filter_args = {}
        self.split_strategy = None
        self.filter_strategy = None
        self.reloaded = False
        self.split_df = None
        self.new_split_entry = None
        self.combined_split_hash = None
        self.class_map = None
        self.baseline_split_df = None

        self.image_overview_df = self._init_image_overview_df(image_overview=image_overview)
        self._init_split_artefacts_path(artefacts_folder=artefacts_folder, image_overview=image_overview)
        self._init_split_overview(split_overview=split_overview)

    def run(
        self,
        split_strategy,
        filter_strategy,
        split_args: Optional[dict] = None,
        filter_args: Optional[dict] = None,
        split_hash: Optional[str] = None,
        baseline_split: Optional[str] = None,
    ) -> pd.DataFrame:

        self.split_strategy = split_strategy
        self.filter_strategy = filter_strategy

        # If a baseline split is requested, load it
        if baseline_split:
            logger.info(f"Loading baseline split: {baseline_split}")
            self._load_baseline_split(baseline_split)
            

        # Prepare split and filter arguments
        self._prepare_split_args(filter_args, split_args)

        # Try to reload using split hash if provided
        if split_hash:
            try:
                match = split_search.find_with_existing_hash(self, split_hash)
                self._reload_split(match)
                self.reloaded = True
                return self.split_df
            except ValueError:
                logger.info("Hash not found, falling back to argument-based matching.")

        # Search for existing split based on arguments
        if self.split_overview is None or self.split_overview.empty:
            logger.warning("Since the split overview was not provided or is empty , a new split and split overview will be created.")
            existing = pd.DataFrame()
        else:
            existing = self.search_for_existing_split_entry()
        
        # Check if an existing split entry was found to start reloading
        if not existing.empty:
            reloaded_df = self._reload_split(existing)
            
            self.split_df = reloaded_df
            self.reloaded = True
            return self.split_df

        # Since no existing split was found, proceed with creation of new split
        self.split_df = self.create_new_split()
        return self.split_df
    
    def create_new_split(self):
        """ Orchestrates the split creation process by applying pre-filtering, filtering, splitting, and post-splitting steps."""
        
        logger.info("No existing split found, creating new one.")
        pre_filtered_df =  self.pre_filtering(self.image_overview_df)
        filtered_df = self.filter_data(pre_filtered_df)
        self.post_filtering(filtered_df)

        pre_split_df =  self.pre_split(filtered_df)
        split_dict = self.split(pre_split_df)
        post_split_df = self.post_split(split_dict)

        split_df = self.finalize_split(post_split_df)
        self.split_checks(split_df)
        return split_df

    
    def _generate_split_hash(self, split_df: pd.DataFrame) -> str:
        """Generate hashes for the split data."""
        try:
            split_hashes = HashGenerator.generate_hash_dict_from_split(
                split_df, col_to_hash="split"
            )
            return HashGenerator.sha256_from_list(split_hashes.values())
        except Exception as e:
            logger.error("Hash generation failed: %s", e)
            raise ValueError("Hash generation failed.") from e

    def _generate_row_to_append(self, description: str = None):
        """
        Appends the split metadata to the split overview DataFrame.

        Args:
            Description: Optional, a description to be added to the split inside the split_overview.
        """
        prefixed_split_args = {
            f"split_{key}": value for key, value in self.split_args.items()
        }
        prefixed_filter_args = {
            f"filter_{key}": value for key, value in self.filter_args.items()
        }

        args_to_process = {
            **prefixed_split_args,
            **prefixed_filter_args,
        }

        args_to_save = self.preprocess_args_for_split_entry(args_to_process)

        # mask empty values to empty strings
        args_to_save = {k: (v if v else "") for k, v in args_to_save.items()}

        new_entry = {
            "split_strategy": self.split_strategy,
            "filter_strategy": self.filter_strategy,
            "combined_split_hash": self.combined_split_hash,
            "description": description,
            **args_to_save 
        }

        print(f"New split entry: {new_entry}")
        self.new_split_entry = pd.DataFrame([new_entry])

    def _prepare_split_path(self, hash_value: str):
        """Prepare the file path where the split will be saved."""
        split_dir = self.artefact_folder / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir / f"{hash_value[:24]}.csv"

    def save_outputs(self, description: Optional[str] = None):
        """
        Saves the split DataFrame and appends its metadata to the split overview.
        """
        if self.reloaded:
            logger.info("Since the split was reloaded, no new artefacts will be saved.")
            return

        # Generate hash and metadata
        self.combined_split_hash = self._generate_split_hash(self.split_df)
        self._generate_row_to_append(description)

        # Save split CSV
        output_path = self._prepare_split_path(self.combined_split_hash)
        self.split_df.to_csv(output_path, index=False)
        logger.info("Saved split to: %s", output_path)
        # Save / update overview CSV
        self.split_overview = (
            self.new_split_entry
            if self.split_overview is None
            else pd.concat([self.split_overview, self.new_split_entry], ignore_index=True)
        )


        self.split_overview.to_csv(self.split_overview_path, index=False)
        logger.info("Updated split overview: %s", self.split_overview_path)



    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the filter strategy to the image overview DataFrame."""
        return apply_filter_by_name(
            filter_name=self.filter_strategy,
            image_overview=df,
            filter_args=self.filter_args,
        )
    
    def split(self, df: pd.DataFrame):
        """Performs the split operation on the image overview DataFrame."""
        split_dict =  perform_splitting(
            split_strategy_function=self.split_strategy,
            image_overview=df,
            split_args=self.split_args,
        )

        split_df  = pd.concat(
            [
                # Concatenate the image and target class to a DataFrame
                pd.concat([image, target_class], axis=1, names=["image", "y_label"])
                # assign the split name to the DataFrame
                .assign(split=split_name)
                # iterate through the dictionary
                for split_name, (image, target_class) in split_dict.items()
            ],
            ignore_index=True,
        )

        return df.merge(split_df[["image", "split"]], on="image")

    def _reload_split(self, row: pd.DataFrame):
        """Reloads an existing split based on the row of the split overview DataFrame.

        Args:
            row: A DataFrame row containing the split information to reload.

        Returns:
           The split DataFrame based on the given hash value.
        """
        # Logic to reload the split based on the row is implemented in the split_reload module.
        reload_logic(self, row)

        
    def search_for_existing_split_entry(self) -> pd.DataFrame:
        """Searches for an existing split based on the current split and filter arguments."""
        
        columns_to_ignore=["combined_split_hash", "description", "ground_test"]
        return split_search.find_existing_split_entry(self, columns_to_ignore)


    def _load_baseline_split(self, reload_key: str):
        if self.split_overview is None:
            raise ValueError("Cannot resolve baseline split — split overview not loaded.")

        match = self.split_overview[
            self.split_overview["description"] == reload_key
        ]

        if match.empty:
            raise ValueError(f"No split found with description '{reload_key}'")

        logger.info(f"Loading baseline split with description: {reload_key}")
        self.baseline_split_df = reload_logic(self, match)


    # === Inputs   ===
    
    def _init_split_artefacts_path(
        self,
        split_folder: Union[str, None] = None,

        artefacts_folder: Union[str, None] = None,
        image_overview = None,
    ):
        """Prepare the paths based on the given input."""

        # check if the artefacts_folder is given and create if not found
        if artefacts_folder is not None:
            pathlib.Path(artefacts_folder).mkdir(parents=True, exist_ok=True)
            self.artefact_folder = pathlib.Path(artefacts_folder)

        # if the image_overview is a string path, extract the parent folder and set it as the artefact_folder
        elif isinstance(image_overview, (str, pathlib.Path)):
            self.artefact_folder = pathlib.Path(image_overview).parent

        # set the default artefact_folder if no artefact_folder is given
        else:
            self.artefact_folder = pathlib.Path(".") / "data" / "split_artefacts"

        # check if the split_folder is given; otherwise, use the default path
        if split_folder is None:
            self.split_folder = self.artefact_folder / "splits"
        else:
            self.split_folder = split_folder

        pathlib.Path(self.split_folder).mkdir(parents=True, exist_ok=True)

    def _init_split_overview(self, split_overview: Union[str, None] = None,):
        # check if the split_overview is given else use the default path
        if split_overview is None:
            self.split_overview_path = self.artefact_folder / "split_overview.csv"

            try: 
                self.split_overview = pd.read_csv(self.split_overview_path).fillna("")
                logger.info("Loaded split overview from: %s", self.split_overview_path)
            except FileNotFoundError:
                self.split_overview = None
                
        elif isinstance(split_overview, pd.DataFrame):
            logger.info("Using provided split overview DataFrame.")
            self.split_overview = split_overview
            self.split_overview_path = self.artefact_folder / "split_overview.csv"

        elif isinstance(split_overview, (str, pathlib.Path)):
            if os.path.exists(split_overview):
                self.split_overview = pd.read_csv(split_overview)
                self.split_overview_path = split_overview
                logger.info("Loaded split overview from: %s", self.split_overview_path)
            else:
                raise FileNotFoundError(
                    f"Split overview path given but no file found: {split_overview}"
                )
    

    def _init_image_overview_df(
        self, image_overview = None
    ) -> pd.DataFrame:
        """Initializes the image overview DataFrame based on the given input.

        Args:
            image_overview_df:
                Can be one of the following:
                - a string path to a CSV file containing the image overview data.
                - a DataFrame containing the image overview.
                - an ZooLakeOverviewCreator instance to extract the image overview from

        Returns:
            A DataFrame containing the image overview data.

        Raises:
            FileNotFoundError: If the loading of the image overview would fail because
                                the file could not be found.
        """

        if isinstance(image_overview, BaseOverviewCreator):
            return image_overview.get_overview_df()

        elif isinstance(image_overview, pd.DataFrame):
            return image_overview

        # check if the overview_df is a string path or a DataFrame
        elif isinstance(image_overview, (str, pathlib.Path)):

            # check if the file exists
            if not os.path.exists(image_overview):
                logger.error("Image overview file not found: %s", image_overview)
                raise FileNotFoundError(
                    f"Image overview file not found: {image_overview}"
                )

            # load the overview_df and fill nan values with an empty string
            return pd.read_csv(image_overview, dtype=str).fillna("")
        

    
        image_overview = self.artefact_folder / "overview.csv"

        if not os.path.exists(image_overview):
            raise FileNotFoundError(
                f"No image overview defined, please provide a image overview or ensure \
                the image overview is inside the artefacts folder with the default name 'overview.csv'")

        return pd.read_csv(image_overview)
    

    def pre_filtering(self, df: pd.DataFrame):
        """Pre-filtering step before applying the filter strategy."""
        return df

    def post_filtering(self, df: pd.DataFrame): 
        """Post-filtering step after applying the filter strategy."""
        return df
    
    def pre_split(self, df: pd.DataFrame):
        """Pre-split step before applying the split strategy."""
        return df

    def post_split(self, df: pd.DataFrame):
        """Post-split step after applying the split strategy."""
        return df
    
    def finalize_split(self, df: pd.DataFrame):
        return df
    
    def split_checks(self, df: pd.DataFrame): 
        """Perform checks on the split df after the split has been performed."""
        pass

    def preprocess_args_for_split_entry(self, args: dict) -> dict:
        """Preprocesses the split and filter arguments for the split entry."""
        return args

    def _prepare_split_args(
            self,
            filter_args: Optional[dict] = None,
            split_args: Optional[dict] = None,
        ): 
        """Prepare and update the split and filter arguments.   
            Args:
            filter_args: Arguments for the filter strategy.
            split_args: Arguments for the split strategy.
        """

        if split_args is None:
            split_args = self.split_args or {}
        if filter_args is None:
            filter_args = self.filter_args or {}
        # apply function to convert the arguments to a list for each value
        filter_args = {
            key: filter_helpers.prepare_args_to_filter(value) for key, value in filter_args.items()
        }
        split_args = {
            key: filter_helpers.prepare_args_to_filter(value) for key, value in split_args.items()
        }

        # Update internal arguments if they have changed
        if split_args != self.split_args:
            self.split_args = split_args

        if filter_args != self.filter_args:
            self.filter_args = filter_args

        logger.info("Split args using for the search of existing split: %s", self.split_args)
        logger.info("Filter args using for the search of existing split: %s", self.filter_args)

    def get_split_overview(self) -> pd.DataFrame:
        """Returns the split overview DataFrame."""
        if self.split_overview is None:
            raise ValueError("Split overview is not initialized.")
        return self.split_overview


    