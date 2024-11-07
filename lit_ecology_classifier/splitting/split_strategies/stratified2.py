from sklearn.model_selection import train_test_split
import logging
from .split_strategy import SplitStrategy


logger = logging.getLogger(__name__)


class Stratified2(SplitStrategy):

    def perform_split(self, df, **kwargs):
        """ Perform a stratified split on the data. 

        Args: Dataframe containing the image names and the class labels
        
        Returns: Dictionary containing the split data. Example: 
                {
                    "train": hash1, hash2, hash3,
                    "val": hash4, hash5, hash6,
                    "test": hash7, hash8, hash
                }
        """

        logger.info("Performing stratified split on the data:%s", df.shape)

        X = df["image"]
        y = df["class"]

        X_train, X_temp, y_train, y_temp  = train_test_split(X,y,
                                                            train_size=0.75, 
                                                            stratify=y,
                                                            random_state=42)

        X_val, X_test, y_val, y_test = train_test_split(
                                                        X_temp, y_temp, 
                                                        test_size=0.5,       
                                                        stratify=y_temp, 
                                                        random_state=42
                                                )
        
        return {
            "train": [X_train, y_train],
            "val": [X_val, y_val],
            "test": [X_test,y_test]
        }

