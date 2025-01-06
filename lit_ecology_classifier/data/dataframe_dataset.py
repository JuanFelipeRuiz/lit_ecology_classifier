import os

from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, Resize, Normalize, ToTensor
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

class DataFrameDataset(Dataset):
    def __init__(self, 
                 image_overview: pd.DataFrame,
                 data_dir: str,
                 train: bool = True, 
                 TTA: bool = False,
                 shuffle: bool = False):
        """ Initialisation of the DataframeDataSet

        Args:
            image_overview: A dataframe containing the image label and iamge name
            train: A bool to identitfy the train dataset
            TTA: A bool to enable test time augementation
            shuffle : Shuffle the data during loading
        """
        self.df = image_overview.copy()
        self.train = train
        self.TTA = TTA
        self.shuffle = shuffle
        self.transforms = {}
        self._define_transforms()

        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def _define_transforms(self):
        """
        Defines the transformations for train, val and test images.
        """
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transforms = {
            "train_transforms": Compose([
                RandomHorizontalFlip(),
                RandomRotation(30),
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean, std),
            ]),
            "val_transforms": Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean, std),
            ]),
        }
        if self.TTA:
            self.transforms["rotations"] = {
                "0": Compose([RandomRotation(0)]),
                "90": Compose([RandomRotation(90)]),
                "180": Compose([RandomRotation(180)]),
                "270": Compose([RandomRotation(270)]),
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx : int):
        """ Retrieves an image and its corresponding label based on the provided index.

        Args:
            idx (int): The index to load

        Returns:
            A dictionary with the image and labael. If TTA is enabled, it returns a dictionary of
            the rotations.
        """
        row = self.df.iloc[idx]
        label = row["label"]
        image_path = os.path.join(label,row["image"])

        # load the image
        image = Image.open(image_path).convert("RGB")

        if self.TTA:
            transformed_images = {
                rotation: self.transforms["val_transforms"](self.transforms["rotations"][rotation](image))
                for rotation in self.transforms["rotations"]
            }
            return {"images": transformed_images, "label": label}

        if self.train:
            image = self.transforms["train_transforms"](image)
        else:
            image = self.transforms["val_transforms"](image)

        return {"image": image, "label": label}
