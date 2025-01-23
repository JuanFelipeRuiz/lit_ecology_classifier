import logging
import os
from typing import Iterable, Union, Optional, Literal

import torch
import pandas as pd
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split

from ..data.imagedataset import ImageFolderDataset
from ..data.tardataset import TarImageDataset
from ..data.dataframe_dataset import DataFrameDataset
from ..helpers.helpers import TTA_collate_fn, setup_classmap

logger = logging.getLogger(__name__)

class DataModule(LightningDataModule):
    """
    A LightningDataModule with PyTorch for handlin training routines compatible with the PyTorch Lightning framework.
    The DataModule is responsible for loading and preparing the data for training, validation, and testing.
    It can handle both image folders and tar files containing the images. A data overview can be provided to
    specify the dataset splits.

    Attributes:
        tarpath (str): Path to the tar file containing the dataset.
        batch_size (int): Number of images to load per batch.
        dataset (str): Identifier for the dataset being used.
        testing (bool): Flag to enable testing mode, which includes TTA (Test Time Augmentation).
        priority_classes (str): Path to the JSON file containing a list of the priority classes.
        splits (Iterable): Proportions to split the dataset into training, validation, and testing.
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        dataset: str,
        TTA: bool = False,
        class_map: dict = {} ,
        priority_classes: list = [],
        rest_classes: list = [],
        splits: Union[Iterable, pd.DataFrame] = [0.7, 0.15],
        **kwargs
    ):
        super().__init__()

        self.datapath = datapath
        self.TTA = TTA  # Enable Test Time Augmentation if testing is True
        self.batch_size = batch_size
        self.dataset = dataset
        if not isinstance(splits, pd.DataFrame):
            self.train_split, self.val_split = splits

        self.split_overview = splits if isinstance(splits, pd.DataFrame) else None
            
        
        if class_map == {} or class_map is None:
            self.class_map = setup_classmap(datapath=datapath, priority_classes=priority_classes, rest_classes=rest_classes)
        else:
            self.class_map = class_map



        self.priority_classes = priority_classes
        self.rest_classes = rest_classes
        self.use_multi = not kwargs.get("no_use_multi", False)
        # Verify that class map exists for testing mode

    def setup(self, stage: Optional[Literal["predict"]] = None):
        """ Set up the correct dataset for the current stage of the model.
        Args:
            stage:  A string indicating the current stage of the model
        """

        if stage != "predict":

            if self.split_overview is not None:
                logger.debug("Setting up a dataset based on asplit overview.")
                self.train_dataset, self.val_dataset, self.test_dataset= self.setup_from_overview()
                logger.info("Train size: %s", len(self.train_dataset))
                logger.info("Validation size: %s", len(self.val_dataset))
                logger.info("Test size: %s", len(self.test_dataset))
                
            else:
                if self.datapath.find(".tar") == -1:
                    logger.debug("Setting up a dataset based on an image folder.")
                    full_dataset = ImageFolderDataset(
                        self.datapath,
                        self.class_map,
                        self.priority_classes,
                        rest_classes=self.rest_classes,
                        TTA=self.TTA,
                        train=True,
                    )
                else:
                    logger.debug("Setting up a dataset based on a tar file.")
                    full_dataset = TarImageDataset(
                        self.datapath,
                        self.class_map,
                        self.priority_classes,
                        rest_classes=self.rest_classes,
                        TTA=self.TTA,
                        train=True,
                    )

                # Since no split overview is provided, create a random split of the dataset
                # This one will not be logged by the split processor
                # Random 

                self.train_dataset, self.val_dataset, self.test_dataset = self.create_random_split(
                        full_dataset = full_dataset
                    )

                # Set the train flag of the validation and test datasets to False
                self.val_dataset.train = False
                self.test_dataset.train = False

        else:
            logger.debug("Setting up dataset for prediction.")
            self.prediction_setup()
           

    def prediction_setup(self):
        """Set up the dataset for prediction by loading the images from the provided directory.
        Can handle tar files and image folders.
        """
        if self.datapath.find(".tar") == -1:
                logger.debug(
                    "Using ImageFolderDataset for prediction, no tar file found."
                )
                self.predict_dataset = ImageFolderDataset(
                    data_dir=self.datapath,
                    class_map=self.class_map,
                    priority_classes=self.priority_classes,
                    rest_classes=self.rest_classes,
                    TTA=self.TTA,
                    train=False,
                )

        else:
            logger.debug("Using tar file for prediction.")
            self.predict_dataset = TarImageDataset(
                    self.datapath,
                    self.class_map,
                    self.priority_classes,
                    self.rest_classes,
                    TTA=self.TTA,
                    train=False,
                )

    def create_random_split(self, 
                         full_dataset: Dataset
                        )-> tuple:
        """Split up the dataset into training, validation, and testing sets based on the provided proportions.

        Args:
            splits: Proportions to split the dataset into training, validation, and testing.
        Returns:
            The different instances of the given Dataset for each split.
        """
        # determine the size of each split based on the provided proportions
        train_size = int(self.train_split * len(full_dataset))
        val_size = int(self.val_split * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        logger.info("Train size: %s", train_size)
        logger.info("Validation size: %s", val_size)
        logger.info("Test size: %s", test_size)

        # Randomly split and initialize the datasets based on the determined sizes
        return random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
    
    def setup_from_overview(self)-> tuple:
        """
        Extract the datasets from the provided split overview

        Args:
            overview: The split overview containing the dataset splits.

        Returns:
            A tuple containing the different splits of the dataset.
        """
        train_df = self.split_overview[ self.split_overview["split"] == "train"].reset_index(drop=True)
        val_df   = self.split_overview[ self.split_overview["split"] == "val"].reset_index(drop=True)
        test_df  = self.split_overview[ self.split_overview["split"] == "test"].reset_index(drop=True)

        logger.info("Train size of df: %s", len(train_df))
        logger.info("Validation size of df: %s", len(val_df))
        logger.info("Test size of df: %s", len(test_df))

        train_dataset = DataFrameDataset(
        image_overview=train_df,
        class_map=self.class_map,
        data_dir=self.datapath,
        train=True,
        TTA=self.TTA
        )

        val_dataset = DataFrameDataset(
            image_overview=val_df,  
            class_map=self.class_map,
            data_dir=self.datapath,
            train=False,
            TTA=self.TTA
        )

        test_dataset = DataFrameDataset(
            image_overview=test_df,
            class_map=self.class_map,
            data_dir=self.datapath,
            train=False,
            TTA=self.TTA
        )

        return train_dataset, val_dataset, test_dataset

    def train_dataloader(self):
        """
        Constructs the DataLoader for training data.
        Returns:
            DataLoader: DataLoader object for the training dataset.
        """


        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=None,
            num_workers= 4,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        """ Constructs a default validation dataloader.

        Returns:
            DataLoader: Default DataLoader object for the validation dataset.
        """

        loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=None,
                num_workers= 4,
                pin_memory=True,
                drop_last=False,
            )
        if self.TTA:
                # Apply TTA collate function if TTA is enabled
                loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    sampler=None,
                    num_workers= 4,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=lambda x:TTA_collate_fn(x,True),
                )
        return loader

    def test_dataloader(self):
        """Constructs a standard dataloader for testing.
        Returns:
            DataLoader: DataLoader object for the testing dataset.
        """

        if self.TTA:
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=None,
                num_workers= 4,
                pin_memory=True,
                drop_last=False,
                collate_fn=lambda x:TTA_collate_fn(x,True),
            )
        else:
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers= 4,
                pin_memory=True,
                drop_last=False,
            )
        return loader

    def predict_dataloader(self):
        """Constructs a default dataloader for inference.

        Returns:
            DataLoader: DataLoader object for the inference dataset.
        """
        loader = DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=None,
                num_workers=8,
                pin_memory=False,
                drop_last=False,
                collate_fn=lambda x:TTA_collate_fn(x,False) if self.TTA else None
            )
    
        return loader


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    import json

    with open("config/rest.json") as file:
        rest = json.load(file)["rest_classes"]
    # Create an instance of the PlanktonDataModule with the specified parameters
    dm = DataModule(
        "./phyto.tar",
        dataset="phyto",
        batch_size=1024,
        testing=False,
        use_multi=False,
        rest_classes=rest,
        splits=[0.7, 0.15],
    )

    # Set up datasets for the 'fit' stage
    dm.setup("fit")
    # Get a DataLoader for training and iterate through it
    test_loader = dm.train_dataloader()
    k = 0
    for i in test_loader:
        print(i[0].shape, len(i[1]))
        k += i[0].shape[0]
    print("number of images", k)
