import os

import pandas as pd

from .create_overview_data_set import CreateOverviewDf

class MoveImagesToIntrim():

    def __init__(self,
                 path_to_overview_dataset : str =None ,
                 created_overview_class : CreateOverviewDf = None,
                 **krawgs,
                 ) -> None:
        
        if created_overview_class == None: 
            created_overview_class = CreateOverviewDf(**kwargs , hash_algorithm = False)

        self.created_overview_class = created_overview_class

        self.overview_df = self._initialize_overview_df(
            created_overview_class,
            path_to_overview_dataset
        )

    def _initialize_overview_df(
            self,
            created_overview_class,
            path_to_overview_dataset
        ) -> pd.DataFrame :
        """Initialisation of overview_df based on the given attributes
        
        Helper function to initialise the overview data frame based on provided path to a csv
        or based on the CreateOverviewDf class
        
        Args:
            created_overview_class (class CreateOverviewDf): Instance of the Class CreateOverviewDf
            path_to_overview_dataset (str): path to the csv containg a overview of the dataframe
        
        Returns:
            Overview_df (pd.Dataframe): A dataframe containg a overview of the image and hashes 

            """


        if path_to_overview_dataset is not None:
            print(f"Lade overview_df von {path_to_overview_dataset}.")
            return self._load_from_path(path_to_overview_dataset)
        elif created_overview_class is not None:
            print("Verwende das bereitgestellte overview_df.")
            return self.created_overview_class.get_overview_df

        

    def _load_from_path(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            raise IOError(f"Error at loading the data{path}: {e}")
        
    def _load_createoverview():
        pass


    def _load_image_overview(self):
        self.created_overview_class.image_paths

    def main(self):
        plankton_classes = self._get_planktonclasses
        pass
    
    def _create_target_folder(self):
        pass
        
    def _get_planktonclasses(self):
        df = self.created_overview_class.get_overview_df()
        return list(df["class"].unique())


        

    def _load_path_overview():
        pass

    def _creeate_interim():
        pass