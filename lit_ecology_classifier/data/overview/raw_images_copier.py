"""
Child class of the base_mover to move the Images from a raw folder to an interim folder.  The class
inherits the main functionalities from the BaseImageMover.
"""

import logging
import os

import pandas as pd

from lit_ecology_classifier.helpers.base_copier import BaseImageCopier
from lit_ecology_classifier.data.overview_creator import OverviewCreator


logger = logging.getLogger(__name__)


class RawImageCopier(BaseImageCopier):
    """Copies the images from the raw folder to the interim folder.
     
    It inherits the main functionalities from the BaseImageCopier. The image paths created 
    by the overview creator are used for the source paths. The target paths are created based 
    on the class of the image.

    Attributes:
        tgt_base_path (str): Destination base path to the folder where the images will be copied.
        overview_creator (overview_creator): Instance of the overview creator class. 
    """

    def __init__(self, tgt_base_path: str, overview_creator: OverviewCreator):
        """
        Initialize the RawImageCopier with the target base path and the OverviewCreator instance.

        Args:
            tgt_base_path (str): Target base path where the images will be copied.
            overview_creator (OverviewCreator): Instance of the OverviewCreator class.
        """
        super().__init__(src_base_path=None, tgt_base_path=tgt_base_path)
        self.overview_creator = overview_creator
        self._images_paths = self._get_image_paths()


    def _get_image_paths(self ) -> dict:
        """Gets the image paths from the created_overview_class attribute "image_paths"

        Returns: a dict with all image paths inside the overview creator. Example
            {
                "1": [path/1/image1, path/1/image2],
                "2": [path/2/image1, path/2/image2],
            }
        """

        return getattr(self.overview_creator, "image_paths")

    def _image_paths_to_df(self, image_paths: dict) -> pd.DataFrame:
        """Converts the image paths from a dict to a pandas DataFrame

        Args:
            image_paths (dict): Dictionary with the image paths. Example:
                {
                    "1": [path/path1/image1, path/path1/image2],..
                    "2": [path/path2/image1, path/path2/image2],..
                }

        Returns: a pandas DataFrame with the image paths and data_set_version.
                | dataset_version | image |
                |-----------------|-------|
                | 1               | path1 |
                | 1               | path2 |

        """

        rows = [
            (label, img_path)
            for label, img_paths in image_paths.items()
            for img_path in img_paths
        ]

        return pd.DataFrame(rows, columns=["data_set_version", "image_path"])

    def _class_finder(self, version: str, datapath) -> str:
        """Finds the class of the image based on the image path and the version of the dataset.

        If the version is "1" the class is the second parent folder of the image path. Example:
            /data/raw/1/training_data/image1.jpg -> class: 1

        Args:
            version (str): Version of the dataset as string.
            datapath (str): Path to the image as string.
        """

        if version == "1":
            return os.path.basename(os.path.dirname(os.path.dirname(datapath)))

        return os.path.basename(os.path.dirname(datapath))

    def _prepare_tgt_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the target paths for the images"""

        df["class"] = df.apply(
            lambda x: self._class_finder(x["data_set_version"], x["image_path"]), axis=1
        )
        df["tgt"] = df["class"].apply(lambda c: os.path.join(self.tgt_base_path, c))
        logger.info("Prepared target paths for the images %s", df.head())
        return df

    def copy_raw_to_interim(self):
        """Moves the images from the raw folder to the interim folder."""

        logger.info("Start preparing paths to copyraw images to interim folder")
        path_dict = self._images_paths

        df = self._image_paths_to_df(path_dict)
        df = self._prepare_tgt_paths(df)
        df["src"] = df["image_path"]

        logger.info("Start moving raw images to interim folder")
        self.copy_images(df)

        logger.info("Finished moving raw images to interim folder")
