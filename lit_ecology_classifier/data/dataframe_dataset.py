"""
Custom dataset to load images images based on the provided dataframe.
Assumes that the dataframe has the following columns:
- image: The image name ending with the extension (e.g. image.jpg)
- label : The label of the image

To use this dataset, the images need to be stored in the following structure:
- data_dir
    - label1
        - image1.jpg
        - image2.jpg
    - label2
        - image3.jpg
        - image4.jpg
"""

import os


from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import AugMix, Compose, Normalize, RandomHorizontalFlip, RandomRotation, Resize, ToDtype, ToImage
from torch.utils.data import Dataset
import pandas as pd

class DataFrameDataset(Dataset):
    def __init__(self, 
                 image_overview: pd.DataFrame,
                 data_dir: str,
                 train: bool = True, 
                 TTA: bool = False,
                 shuffle: bool = False,
                 class_map: dict = None,):
        """ Initialisation of the DataframeDataSet

        Args:
            image_overview: A dataframe containing the image label and iamge name
            train: A bool to identitfy the train dataset
            TTA: A bool to enable test time augementation
            shuffle : Shuffle the data during loading
        """
        self.df = image_overview.copy()
        self.data_dir = data_dir
        self.train = train
        self.TTA = TTA
        self.shuffle = shuffle
        self.transforms = {}
        self._define_transforms()
        self.class_map = class_map 


        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def _define_transforms(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet mean and std #TODOchange it back to 30
        self.train_transforms = Compose([ToImage(), RandomHorizontalFlip(), RandomRotation(180), AugMix(severity=6,mixture_width=5), Resize((224, 224)), ToDtype(torch.float32, scale=True), Normalize(mean, std)])
        self.val_transforms = Compose([ToImage(), Resize((224, 224)), ToDtype(torch.float32, scale=True), Normalize(mean, std)])
        if self.TTA:
            self.rotations = {
                "0": Compose([RandomRotation(0, 0)]),
                "90": Compose([RandomRotation((90, 90))]),
                "180": Compose([RandomRotation((180, 180))]),
                "270": Compose([RandomRotation((270, 270))]),
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
        label = row["class"]
        class_map = row["class_map"]
        image_path = os.path.join(self.data_dir ,label,row["image"])

        # load the image
        image = Image.open(image_path).convert("RGB")

        if self.TTA:
                image = {rot: self.val_transforms(self.rotations[rot](image)) for rot in self.rotations}
        elif self.train:
            image = self.train_transforms(image)
        else:
            image = self.val_transforms(image)
        return image, class_map
