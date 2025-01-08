"""
Main module to ensure reproducibility and  all necessary functions to check for existing splits.
Contains the logic to:
- check for existing splits based on the given arguments or based on the hash value.
- reload a specific split based on the given arguments or hash value
- orcheestrate the creation of a new split based on the given split- and filter strategy
    with the help of the filter and split manager.
"""

import logging
import os

import pandas as pd

from lit_ecology_classifier.splitting.split_strategies.base_split_strategy import BaseSplitStrategy
from lit_ecology_classifier.splitting.filtering.base_filter import BaseFilter
from lit_ecology_classifier.splitting.split_strategies.split_manager import SplitManager
from lit_ecology_classifier.helpers.hashing import HashGenerator
import lit_ecology_classifier.helpers.helpers as helpers
from lit_ecology_classifier.data_overview.overview_creator import OverviewCreator
from lit_ecology_classifier.splitting.filtering.filter_manager import FilterManager


logger = logging.getLogger(__name__)


class SplitProcessor:
    """
    Process the  the checking of existing splits with used arguments and orchestrate the creation of new splits 
    with the help of the filter and split manager modul. 

    It gives the possibility to search for existing splits based on the given arguments or based on the hash value.
    The hash value is used a unique identifier for the split and is generated based on the images hash values grouped
    by each split and hashed together.

    Attributes:
        split_df: 
            DataFrame containing the split data with the image names, class labels and the split
            labels. 
        image_overview_df:
            A DataFrame containing the overview of the images to be split. It needs to contain
            at least the columns "image", "class" and a hash value of the images.
        split_overview_df:
            A DataFrame containing the overview of created splits. It needs to contain data
            on the used split and filter strategies, the used arguments and the combined split 
            hash.
        split_outpath:
            Paths to save the split overview and the split data.
        split_strategy:
            The split strategy to be used for the split process. Can be a custom instance of a
            SplitStrategy or a class name to use already implemented split strategies. To ensure
            the correct reproducibility of the split, the given 
            It is better
            to give a already initialized instance.
        split_args:
            Arguments to be passed to the split strategy.
        filter_strategy:
            Instance of the filter strategy to filter the image overview before the split process.
            Initialized with the given filter arguments.
        filter_args:
            Arguments to be passed to the filter strategy.
        class_map:
            Dictionary containing the class mapping of the class labels. 
        priority_classes:
            List of class labels to be prioritized in the class mapping.
        rest_classes:
            List of class labels to keep in the class mapping. All other classes that are neither
            in the priority_classes nor in the rest_classes are removed from the class mapping and
            not used for the further split and training process

    Methods:
        search_splits: 
            Search for existing splits based on the given arguments or based on the hash value.
            Will be executed at the initialization of the SplitProcessor and can be re-executed 
            to search or create new splits based on the given arguments. OOnly one split will 
            be available at the same time.

    """

    _FilterManager =  FilterManager
    _SplitManager = SplitManager
    
    def __init__(
        self,
        image_overview: str | pd.DataFrame | OverviewCreator,
        split_overview: str | pd.DataFrame = None,
        split_hash: str = None,
        split_folder: str = None,
        split_strategy: BaseSplitStrategy | str = None,
        split_args: dict = None,
        filter_strategy: BaseFilter | str = None,
        filter_args: dict = None,
        class_map: dict[str, int] = None,
        rest_classes: list[str] = None,
        priority_classes: list[str] = None,
    ):
        """Initialisation of the SplitProcessor.

        Args:
            image_overview:
                String, DataFrame or OverviewCreator instance  containing the image overview with
                the image, class labels, image hashes and other relevant informations of the images
                to be split.
            split_overview:
                Optional, a DataFrame or string path to a CSV file containing the split overview
                with the split strategy, filter strategy, arguments and the combined split hash to
                check and reload existing splits.
            split_hash:
                Optional, a hash value to reload a specific split based on the hash value.
            Split_folders:
                Optional, path to the folder with the split data. 
            split_strategy:
                Split strategy to be used for the split process. Can be a custom instance of a
                SplitStrategy or a class name to use already implemented split strategies.
                Example: "random_split"
            split_args:
                Arguments to be passed to the split strategy. Example: {"test_size": 0.7}
            filter_strategy:
                Filter strategy to filter the image overview before the split process. Can be a
                custom instance of a FilterStrategy or a class name to use already implemented
                filter strategies. Example: "plankton_filter"
            filter_args:
                Arguments to be passed to the filter strategy. Example: {"dataset_version": "v1"}
            class_map:
                Optional, dictionary containing the class mapping of the class labels. If no class
                mapping is provided, the class mapping is extracted from the image overview.
            priority_classes:
                Optional, list of class labels to be prioritized in the class mapping.
            rest_classes:
                Optional, list of class labels to keep in the class mapping. All other classes that 
                are neither in the priority_classes nor in the rest_classes are removed from the 
                class mapping. And not used for the futher split and training process.
        """
        self.split_folder = split_folder

        self.class_map = class_map if class_map is not None else {}
        self.rest_classes = rest_classes if rest_classes is not None else []
        self.priority_classes = priority_classes if priority_classes is not None else []

        # Check if needed data is provided as DataFrame or as path to a CSV file
        self.image_overview_df = self._init_image_overview_df(image_overview)
        self.split_overview_df = self._init_split_overview_df(split_overview)

        self.filter_strategy = filter_args 
        self.split_strategy = split_args 
        self.filter_args = None
        self.split_args = None

        self.split_df = self.search_splits(
            split_strategy=split_strategy,
            filter_strategy=filter_strategy,
            hash_value=split_hash,
            filter_args=self.filter_args,
            split_args=self.split_args,
        )

        self.row_to_append = None

    def _init_image_overview_df(
        self, image_overview: pd.DataFrame = None | str | OverviewCreator
    ) -> pd.DataFrame:
        """Initializes the image overview DataFrame based on the given input. 
       
        Args:
            image_overview_df:
                Can be one of the following:
                - a string path to a CSV file containing the image overview data.
                - a DataFrame containing the image overview.
                - an OverviewCreator instance to extract the image overview from

        Returns:
            A DataFrame containing the image overview data.
        
        Raises:
            FileNotFoundError: If the loading of the image overview would fail because
                                the file could not be found.
        """

        # check if the overview_df is a string path or a DataFrame
        if isinstance(image_overview, str):

            # check if the file exists
            if not os.path.exists(image_overview):
                logger.error("Image overview file not found: %s", image_overview)
                raise FileNotFoundError(
                    f"Image overview file not found: {image_overview}"
                )

            # load the overview_df
            image_overview = pd.read_csv(image_overview)

        elif isinstance(image_overview, OverviewCreator):   
            image_overview = image_overview.get_overview_df()

        logger.debug("Image overview column types: %s", image_overview.dtypes)
        return image_overview

    def _init_split_overview_df(
        self, split_overview: str | pd.DataFrame
    ) -> pd.DataFrame:
        """Initializes the split overview DataFrame.

        Detects if a split overview DataFrame is provided or a string path to a CSV file.

        Args:
            split_overview: Can be one of the following:
                - a string path to a CSV file containing the split overview data.
                - a DataFrame containing the split overview.

        Returns:
            The split overview DataFrame

        Raises:
            FileNotFoundError: If the loading of the split overview would fail because
                                the file could not be found.
        """

        if split_overview is None:
            logger.info(
                "No split overview provided, no search for existing splits possible"
            )
            return None

        if isinstance(split_overview, str):
            if not os.path.exists(split_overview):
                logger.error("Split overview file not found: %s", split_overview)
                raise FileNotFoundError(
                    f"Split overview file not found: {split_overview}"
                )
            split_overview = pd.read_csv(split_overview)
        logger.info("Split overview loaded.")
        return split_overview
    
    def _ensure_class_name(self, input_: str | BaseFilter | BaseSplitStrategy) -> str:
        """Ensures the Class name is written in CamelCase for the comparison.

        Args:
            input_: Can be a string, an instance of BaseFilter or SplitStrategy 
            to be checked.

        Returns:
            A string representation of the given input in CamelCase.
        
        Raises:
            A ValueError if the given input is in snake_case or starts in lower case.
        """
        # extract class name if the given string is an instance of BaseFilter or SplitStrategy
        if isinstance(input_, (BaseFilter, BaseSplitStrategy)):
            input_ = input_.__class__.__name__

        # check if the given string is a string, else raise an error
        if not isinstance(input_, str):
            raise ValueError(
                "The given string has to be a string or an instance of BaseFilter or SplitStrategy"
            )

        # check if the string is written in CamelCase
        if "_" in input_ or input_.islower():
            raise ValueError(
                "The given string has to be written in CamelCase for the comparison. Please rename the input: %s",
                input_
            )
        
        logger.debug("Class name: %s", input_)
        return input_

    def _prepare_filepath(self, hash_value: str) -> str:
        """Prepares the file path based on the provided hash.

        Args:
            hash_value: A hash value to be used for the file name.

        Returns:
            The file path based on the hash value and given split outputpath.
        """
        filename = f"{str(hash_value)[:24]}.csv"


        filepath = os.path.join(self.split_folder, filename)
        return filepath

    def _reload_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reloads an existing split based on the row of the split overview DataFrame.

        Args:
           df: A DataFrame with one row containing the split to be reloaded.

        Returns:
           The split DataFrame based on the given hash value.
        """

        hash_value = df["combined_split_hash"].values[0]
        logger.info("Reloading split based on hash value: %s", hash_value)  
        
        filepath = self._prepare_filepath(hash_value)
        logger.info("Split file path: %s", filepath)
        if not os.path.exists(filepath):
            logger.error("Split file not found:%s", filepath)
            raise FileNotFoundError(f"Split file not found: {filepath}")
        
        return pd.read_csv(filepath_or_buffer = filepath)
    
    def _find_with_existing_hash(self, hash_value: str) -> pd.DataFrame:
        """Finds a split based on the given hash value.
        
        Args:
            hash_value: Hash value to reload an existing split.
        
        Returns:
           The split DataFrame based on the given hash value.
        """
        existing_split = self.split_overview_df[
            self.split_overview_df["combined_split_hash"] == hash_value
        ]
        if existing_split.empty:
            raise ValueError("No split found with the given hash value.")
        return self._reload_split(df=existing_split)
    

    def _find_existing_split(self, filter_strategy, split_strategy) -> pd.DataFrame:
        """Find an existing split based on filter and split strategy
        
        Args:
            filter_strategy: Filter strategy to be used for the split.
            split_strategy: Split strategy to be used for the split.
        
        Returns:
            A DataFrame containing rows that match the given filter and split strategy.
        """
        return self.split_overview_df[
            (self.split_overview_df["filter_strategy"] == self._ensure_class_name(filter_strategy)) &
            (self.split_overview_df["split_strategy"] == self._ensure_class_name(split_strategy))
        ]

    def _arguments_match(self, existing_split: pd.DataFrame) -> bool:
        """Check if the arguments match the existing split
        
        Args:
            existing_split: DataFrame containing the prefiltered split overview
            based on the filter and split strategy.
        
        Returns:
            A boolean value indicating if the arguments match the existing split.
        """
        columns_to_check = list(self.split_args.keys()) + list(self.filter_args.keys())

        for column in columns_to_check:
            if column in self.split_overview_df.columns:
                if self.split_overview_df[column].iloc[0] != existing_split[column].iloc[0]:
                    return False
        return True
    
    def _merge_class_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the class map to the DataFrame
        
        Args:
            df: DataFrame to merge the class map to.
        
        Returns:
            A DataFrame containing the merged class map.
        """
        try:
            df = df.set_index("class")
            class_map_df = pd.DataFrame(self.class_map.items(), columns=["class", "class_map"]).set_index("class")

           
            return df.merge(class_map_df, left_index=True, right_index=True).reset_index()

        except Exception as e:
            logger.error("Merging class map into filtered DataFrame failed: %s", e)
            raise ValueError("Merging class map failed.") from e
        
    def _generate_class_map(self, df: pd.DataFrame) -> dict[str, int]:
        """Generate a class map based on the given DataFrame
        
        Args:
            df: DataFrame containing the class labels.
        
        Returns:
            A dictionary containing the class mapping of the class labels.
        """

        if self.class_map is None or not self.class_map:
            logger.info("No class map provided, generating class map.")
            self.class_map =  helpers.extract_class_mapping_df(df = df,
                )
        
        logger.info("Class map generated: %s", self.class_map )

        self.class_map = helpers.filter_class_mapping(
            self.class_map, self.priority_classes, self.rest_classes
        )

        logger.info("Class map generated: %s", self.class_map)

    def _generate_split_hash(self, split_df: pd.DataFrame) -> str:
        """Generate hashes for the split data."""
        try:
            split_hashes = HashGenerator.generate_hash_dict_from_split(
                split_df, col_to_hash="split"
            )
            self.combined_split_hash = HashGenerator.sha256_from_list(split_hashes.values())
        except Exception as e:
            logger.error("Hash generation failed: %s", e)
            raise ValueError("Hash generation failed.") from e

    def _generate_row_to_append(self, description: str = None):
        """
        Appends the split metadata to the split overview DataFrame.

        Args:
            combined_split_hash: A hash value to be used for the file name.

        Returns:
        """
        new_entry = {
            "split_strategy": self.split_strategy.__class__.__name__,
            "filter_strategy": self.filter_strategy.__class__.__name__,
            "combined_split_hash": self.combined_split_hash,
            "description": description,
            "class_map": self.class_map,
            **self.split_args,
            **self.filter_args,
        }

        return pd.DataFrame(new_entry, index=[0])
        

    def _new_split(self, split_strategy, filter_strategy) -> pd.DataFrame:
        """Create a new split based on the given split and filter strategy
        
        Args:
            split_strategy: Split strategy to be used for the split.
            filter_strategy: Filter strategy to be used for the split.
        
        Returns:
            A DataFrame containing the split data based on the given split and filter strategy.
        """

        logger.info(
            "Creating a new split based on the given split and filter strategy.:%s", 
                    type(self.image_overview_df)
            )
        
        # Filter the image overview
        filter_manager = FilterManager(filter_strategy=filter_strategy, filter_args= self.filter_args)
        filtered_df = filter_manager.apply_filter(self.image_overview_df)

        # Generate the class map
        self._generate_class_map(filtered_df)

        filtered_df = self._merge_class_map(filtered_df)

        split_manager = SplitManager(
            split_strategy=split_strategy, split_args=self.split_args
        )

        split_df = split_manager.perfom_split(filtered_df)

        # join the split data with the filtered df to keep the metadata
        self.split_df = filtered_df.merge(split_df, on="image")

        self.combined_split_hash = self._generate_split_hash(self.split_df)
        self._generate_row_to_append()
        return self.split_df 

    def search_splits(
        self,
        filter_strategy: str | BaseFilter = None,
        filter_args: dict = None,
        split_strategy: str | BaseSplitStrategy = None,
        split_args: dict = None,
        hash_value: str = None,
    ) -> pd.DataFrame:
        """
        Search for existing splits based on the given split- and filter strategy 
        or based on the hash_value.

        Args:
            filter_strategy: String or instance of the filter module to search for.
            filter_args: Arguments for the filter strategy.
            split_strategy: String or instance of the split strategy to search for.
            split_args: Arguments for the split strategy.
            hash_value: Hash value to reload an existing split.

        Returns:
            pd.DataFrame: The split DataFrame based on the given arguments.
        """
        # Reload split based on hash value
        if hash_value:
            return self._find_with_existing_hash(hash_value)

        # Initialize split_args and filter_args if not provided
        split_args = split_args or {}
        filter_args = filter_args or {}

        # Update internal arguments if they have changed
        if split_args != self.split_args or filter_args != self.filter_args:
            self.split_args = split_args
            self.filter_args = filter_args
            logger.debug("Updated split args: %s", self.split_args)
            logger.debug("Updated filter args: %s", self.filter_args)

        # Check if the split overview exists
        if self.split_overview_df is None:
            logger.info("No split overview found, creating a new split.")
            return self._new_split(split_strategy=split_strategy, filter_strategy=filter_strategy)

        # Filter existing splits based on strategies
        existing_split = self._find_existing_split(filter_strategy, split_strategy)

        if not existing_split.empty:
            # Validate arguments against existing splits
            if self._arguments_match(existing_split):
                logger.info("Existing split found, reloading split.")
                return self._reload_split(existing_split)

            logger.info("Arguments do not match, creating a new split.")
            return self._new_split(split_strategy=split_strategy, filter_strategy=filter_strategy)

        logger.info("No existing split found, creating a new split.")
        return self._new_split(split_strategy=split_strategy, filter_strategy=filter_strategy)
