"""
Module to move images based on a split DataFrame. It inherits the main functionalities from the BaseImageMover class (image_mover.py).
Build the target and source paths based on therow values and provided base paths.
"""

import os
import pandas as pd
import logging

from lit_ecology_classifier.data.mover.base_mover import BaseImageMover

logger = logging.getLogger(__name__)


class SplitImageMover(BaseImageMover):
    """
    Mover class to move images based on a split DataFrame. It inherits the main functionalities
    from the BaseImageMover class.

    Inherits from:
        BaseImageMover (BaseImageMover): Base class for methods to create, copy, and clean folders for images.
    """

    def __init__(self, src_base_path: str, tgt_base_path: str):
        """
        Initialize the SplitImageMover with source and destination base paths and a split DataFrame.

        Args:
            src_base_path (str): Base path to the folder containing the class folder and images.
                                Exampel: /data/interim

            tgt_base_path (str): Base path to the folder containing the class folder and images.
                                 Exampel: /data/proccessed

        """
        super().__init__(src_base_path, tgt_base_path)

    def _prepare_src_path(self, df: pd.DataFrame):
        """
        Prepare the source path by joining the base path, class and image name for each row in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the image names, class and the corresponding split.

        Returns:
            (pd.DataFrame): DataFrame with src paths as column
        """
        df["src"] = df.apply(
            lambda x: os.path.join(x["class"], x["image"]), axis=1
        )
        return df

    def _prepare_tgt_path(self, df: pd.DataFrame):
        """
        Prepare the target path by joining the base path, split, class and image name for each row in the DataFrame.
        Example of the dataframe transformation:
        | class | image | split | tgt |
        |-------|-------|-------|-----|
        | class1| img1  | train | /data/processed/train/class1/img1 |

        Args:
            df (pd.DataFrame): DataFrame containing the image names, class and the corresponding split.

        Returns:
            df (pd.DataFrame): DataFrame containing the target path for each image.

        """
        df["tgt"] = df.apply(
            lambda x: os.path.join(
                self.tgt_base_path, x["split"], x["class"]
            ),
            axis=1,
        )
        return df

    def copy_images(self, df: pd.DataFrame):
        """
        Move images from the source to the target path based on the split DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the image names, class and the corresponding split.

        Returns:
            None
        """

        # prepare the source path columns
        df = self._prepare_src_path(df)

        # prepare the target path column 
        df = self._prepare_tgt_path(df)
        
        # filter columns 
        df = df[["src", "tgt"]]

        # move images
        super().move_images(df)
        return None
