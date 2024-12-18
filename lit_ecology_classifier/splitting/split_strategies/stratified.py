"""


"""

import logging

from sklearn.model_selection import train_test_split

from lit_ecology_classifier.splitting.split_strategies.base_split_strategy import BaseSplitStrategy


logger = logging.getLogger(__name__)


class Stratified(BaseSplitStrategy):
    """
    """

    def __init__(self, train_size = 0.75, test_size = 0.5):
        
        self.train_size = train_size
        self.test_size = test_size

    def perform_split(self, df,  y_col= "class"):
        """ Perform a stratified split on the data. 

        Args: Dataframe containing the image names and the class labels
        
        Returns: Dictionary containing the split data. Example: 
                {
                    "train": image_1, 
                    "val": [X_val, y_val],
                    "test": [X_test,y_test]
                }
        """

        logger.info("Performing stratified split. Shape of data:%s", df.shape)

        X = df["image"]
        y = df[y_col]

        X_train, X_temp, y_train, y_temp  = train_test_split(X,y,
                                                            train_size = self.train_size, 
                                                            stratify = y,
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

