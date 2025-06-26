import logging
from typing import Union, Optional
import pathlib
import pandas as pd

from lit_ecology_classifier.splitting.base_split_processor import BaseSplitProcessor
from lit_ecology_classifier.splitting.split_strategies.split_resolver import perform_splitting
from lit_ecology_classifier.data_overview.zoolake_overview import ZooLakeOverviewCreator
from lit_ecology_classifier.helpers import helpers

logger = logging.getLogger(__name__)

class SplitProcessor(BaseSplitProcessor):
    """Implements BaseSplitProcessor with specific plankton data handling."""
    
    def __init__(
        self,
        split_overview: Optional[Union[str, pathlib.Path, pd.DataFrame]] = None,
        artefacts_folder: Optional[Union[str, pathlib.Path]] = None,
        image_overview: Optional[Union[str, pathlib.Path, pd.DataFrame, ZooLakeOverviewCreator]] = None,
    ):
        
        super().__init__(
            split_overview=split_overview,
            artefacts_folder=artefacts_folder,
            image_overview=image_overview
        )

    def create_class_map(self):
        """Create a class map from the image overview."""
        classes = self.image_overview_df['class'].unique()
        self.class_map = {cls: idx for idx, cls in enumerate(classes)}
        
        logger.info("Class map created: %s", self.class_map)

    def pre_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate class map before filtering."""
        # Extract class-related args from filter_args

        self.create_class_map()

        self.rest_classes = self.filter_args.get('rest_classes', [])
        self.priority_classes = self.filter_args.get('priority_classes', [])
        
        self.class_map = helpers.filter_class_mapping(
            self.class_map,
            priority_classes=self.priority_classes,
            rest_classes=self.rest_classes,
        )    

        return df


    def post_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply class mapping after filtering."""
        self.class_map = helpers.filter_class_mapping(
            self.class_map,
            priority_classes=self.priority_classes,
            rest_classes=self.rest_classes,
        )
        logger.info("Class map after filtering: %s", self.class_map)
        return self._merge_class_map(df)

    def pre_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for split by merging the baseline if available."""
        if self.baseline_split_df is not None:
            logger.info("Merging baseline split DataFrame into image overview.")
            df = df.merge(
                self.baseline_split_df[["hash", "split"]],
                on="hash",
                how="left",
            )

            # check if they are new images that are not in the baseline split
            if df["split"].isnull().any():
                logger.warning("Some images are not in the version of the baseline split DataFrame."
                               "Please set the version to filter for the baseline split DataFrame ")
                
            else:
                logger.info("All images found in the baseline split DataFrame.")

        else:
            logger.info("No baseline split DataFrame available, proceeding without merge.")

        

        return df

    def split(self, df: pd.DataFrame):
        """Performs the split operation on the image overview DataFrame."""

        # Here we check if we need to do a split at all.
        if self.baseline_split_df is not None:
            logger.info("Since a baseline split DataFrame is provided, we will just split the images that are not in the baseline split DataFrame.")
            # since the merge happened in pre_split, we can filter the DataFrame to only include images that are not in the baseline split
            df = df[df["split"].isnull()] 

            if df.empty: 
                return df
            # since we have new images that are not in the baseline split, we proceed the split process with the new images only.
            else:
                df = df[df["split"].isnull()].drop(columns=["split"])

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

    def post_split(self, df) -> pd.DataFrame:
        """Join split data with filtered data to keep metadata."""
        if "split" not in df.columns:
            raise ValueError("Split column missing from split result")
        return df

    def get_class_map(self) -> dict:
        """Return the current class mapping."""

        return self.class_map

    def _merge_class_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge class mapping into DataFrame."""
        try:
            class_map_df = pd.DataFrame(
                self.class_map.items(), 
                columns=["class", "class_map"]
            ).set_index("class")

            df = df.merge(
                class_map_df, 
                left_on="class", 
                right_index=True,
                how="left",
            ).reset_index(drop=True)
            df['class_map'] = df['class_map'].fillna('rest')
            return df
        except Exception as e:
            logger.error("Failed to merge class map: %s", e)
            raise ValueError("Class map merge failed") from e
        