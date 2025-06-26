

import logging
import pandas as pd
from lit_ecology_classifier.data_overview.utils.base_overview_creator import BaseOverviewCreator
from lit_ecology_classifier.data_overview.utils.raw_split_preparer import _RawSplitPathPreparer
from lit_ecology_classifier.data_overview.utils.raw_split_applier import _RawSplitApplier
from lit_ecology_classifier.data_overview.utils.image_processing import ProcessImage

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s - %(levelname)s - %(message)s'
)
class ZooLakeOverviewCreator(BaseOverviewCreator):
    """ZooLake-specific implementation of the OverviewCreator with fixed sha256 hashing."""

    def __init__(self, dataset_version_paths=None):
        # Fixed image processor with sha256 hash
        ImageProcessor = ProcessImage()
        super().__init__(dataset_version_paths=dataset_version_paths, ImageProcessor=ImageProcessor)

        self._overview_with_splits_df = None

    def clean_up_raw_dataset(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self._add_one_hot_encoded_versions(df)
        return df

        
    def _helper_group_by_aggregation(self, group: pd.DataFrame, ranked_datasets: list) -> pd.Series:
        """Return the row from the highest-ranked dataset (excluding 'unlabeled').
        
        and set the dataset column to true, for each dataset it appears in the group."""
        base_row = None
        for dataset_col in ranked_datasets:
            if dataset_col in group.columns:
                match = group[group[dataset_col] == 1]
                if not match.empty:
                    base_row = match.iloc[0].copy()
                    break
        
        if base_row is None:
            base_row = group.iloc[0].copy()
        
        # Preserve all dataset flags
        dataset_cols = [col for col in group.columns if col.startswith('dataset_')]
        for col in dataset_cols:
            base_row[col] = group[col].any()
        
        return base_row
        
    def _add_one_hot_encoded_versions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that we can see in which dataset the image appears.

        Args:
            df: DataFrame containing the overview of the ZooLake dataset.
        """
        count_per_dataset_raw = df["dataset"].value_counts()

        # Convert to prefixed form to match one-hot columns
        count_per_dataset = {
            f"dataset_{k}": v for k, v in count_per_dataset_raw.items()
        }

        df_versions = pd.get_dummies(df["dataset"], prefix="dataset", drop_first=False)
        dataset_columns = df_versions.columns.tolist()


    
        dataset_columns = [col for col in dataset_columns if "dataset_unlabelled" not in col.lower() and "dataset_ood" not in col.lower()]

        ranked_datasets = sorted(dataset_columns, key=lambda x: count_per_dataset[x], reverse=True)

        df = pd.concat([df, df_versions], axis=1)


        grouped = df.groupby(["image", "class", "hash"], as_index=False, sort=False
        ).apply(
            lambda group: self._helper_group_by_aggregation(group, ranked_datasets),
            include_groups=False  
        )

        # remove the dataset column 
        grouped = grouped.drop(columns=["dataset"], errors='ignore')

        return grouped.reset_index(drop=True)

    def get_overview_with_splits_df(self, reload=False):
        """Get the overview DataFrame with split info merged in."""
        if self._overview_with_splits_df is None or reload:
            logging.info("Preparing paths to the split overview files.")
            split_paths = _RawSplitPathPreparer(self.dataset_versions_path).prepare_split_paths()
            logging.info("Split paths prepared. Found paths: %s", split_paths)
            overview_df = self.get_overview_df()
            logging.info("Applying splits to the overview DataFrame")
            self._overview_with_splits_df = _RawSplitApplier(split_paths).apply_splits(overview_df)
        return self._overview_with_splits_df

    

if __name__ == "__main__":
    # Example usage
    dataset_paths = {
        "ZooLake1": "data/ZooLake1",
        "mini_dataset": "data/mini_dataset",
        "OOD": "data/OOD/mini_OOD",
    }
    
    creator = ZooLakeOverviewCreator(dataset_version_paths=dataset_paths)
    overview_df = creator.get_overview_with_splits_df()
    overview_df.to_csv("overview_df_debug.csv", index=False)
    print(overview_df.head())