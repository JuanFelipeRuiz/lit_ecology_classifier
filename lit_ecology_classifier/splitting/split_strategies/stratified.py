"""
Stratified split strategy for splitting the data into train, validation and test sets.
Demonstrates the use of the BaseSplitStrategy class.

It returns a dictionary containing the splits sets train, validation and test.
"""

import logging

from sklearn.model_selection import train_test_split

from lit_ecology_classifier.splitting.split_strategies.base_split_strategy import (
    BaseSplitStrategy,
)


logger = logging.getLogger(__name__)


class Stratified(BaseSplitStrategy):
    """
    Stratified split strategy for splitting the data into train, validation and test sets.
    Uses the sklearn train_test_split method. The data is split into train, validation and test sets
    Default split:
        - Training set: 75%
        - Validation set: 12.5%
        - Test set: 12.5%

    Attributes:
        train_size: The size of the training set.
        test_size: The size of the test set based on the remaining data after the training set.
    """

    def __init__(self, train_size=0.75, test_size=0.5):
        """Initialize the Stratified split strategy.

        Args:
            train_size: The size of the training set. Default is 0.75.
            test_size: The size of the test set. Default is 0.5 of the remaining data
                        after the training set.

        """

        self.train_size = train_size
        self.test_size = test_size

    def perform_split(self, df, y_col="class"):
        """Perform a stratified split on the data.

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

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=self.train_size, stratify=y, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        return {
            "train": [X_train, y_train],
            "val": [X_val, y_val],
            "test": [X_test, y_test],
        }
