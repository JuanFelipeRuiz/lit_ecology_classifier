import pandas as pd
from lit_ecology_classifier.data_overview.utils.base_overview_creator import BaseOverviewCreator
from lit_ecology_classifier.data_overview.utils.raw_split_preparer import _RawSplitPathPreparer
from lit_ecology_classifier.data_overview.utils.raw_split_applier import _RawSplitApplier
from lit_ecology_classifier.data_overview.utils.image_processing import ProcessImage


class ZooLakeOverviewCreator(BaseOverviewCreator):
    """ZooLake-specific implementation of the OverviewCreator with fixed sha256 hashing."""

    def __init__(self, dataset_version_paths=None):
        # Fixed image processor with sha256 hash
        ImageProcessor = ProcessImage()
        super().__init__(dataset_version_paths=dataset_version_paths, ImageProcessor=ImageProcessor)

    def attach_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self._add_one_hot_encoded_versions(df)
        #df = self._fix_ood_columns(df)
        return df

        
    def _helper_group_by_aggregation(self, group: pd.DataFrame, ranked_datasets: list) -> pd.Series:
        """Return the row from the highest-ranked dataset (excluding 'unlabelled')."""
        for dataset_col in ranked_datasets:
            if dataset_col in group.columns:
                match = group[group[dataset_col] == 1]
                if not match.empty:
                    return match.iloc[0]
        return group.iloc[0]
        
    def _add_one_hot_encoded_versions(self, df: pd.DataFrame) -> pd.DataFrame:
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


        grouped = df.groupby(
            ["image", "class", "hash"], 
            as_index=False, 
            sort=False
        ).apply(
            lambda group: self._helper_group_by_aggregation(group, ranked_datasets),
            include_groups=False  # Add this parameter to address the warning
        )


        return grouped.reset_index(drop=True)

 



    def get_overview_with_splits_df(self, reload=False):
        """Get the overview DataFrame with split info merged in."""
        if self._overview_with_splits_df is None or reload:
            split_paths = _RawSplitPathPreparer(self.dataset_versions_path).prepare_split_paths()
            overview_df = self.get_overview_df()
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
    overview_df.to_csv("overview_df_Debug.csv", index=False)
    print(overview_df.head())