"""
Provides the main functionalities to copy images from a source directory to a destination directory.
Besides the copy functionality, it provides functionality to create and clean folders for images.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import shutil
from typing import List, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class BaseImageCopier:
    """
    Base class for copying images from a source directory to a destination directory.
    It provides methods to create, copy, and clean folders for images.

    Attributes:
        src_base_path (str): Source base path to the folder containing the class folder and images.
        tgt_base_path (str): Target base path to the folder containing the class folder and images.
    """

    def __init__(self, src_base_path: str = None, tgt_base_path: str = None):
        """
        Initialize the BaseImageCopier with source and destination base paths.

        Args:
            src_base_path (str): Optional base path to the folder containing the
                                 class folders with images.
            tgt_base_path (str): Base path to the terget folder.
        """
        self.src_base_path = src_base_path
        self.tgt_base_path = tgt_base_path

    def create_tgt_folder(self, folder_path: str):
        """
        Creates a folder at the specified target path after checking if it exists.

        The creation of the folder is done with the exist_ok=True parameter to avoid
        train/test errors if the folder already exists.

        Args:
            folder_path (str): Path to the folder to be created.
        """
        try:
            os.makedirs(folder_path, exist_ok=True)
            return None
        except Exception as e:
            logger.error("Error creating the folder %s: %s", folder_path, e)
            raise

    def _prepare_target_folders(self, paths_df: pd.DataFrame):
        """
        Prepare the target folders for the images to be copied.

        Filters all unique target folders and applies the function to create a folder to each one.

        Args:
            paths_df (pd.DatFrame): Dataframe containing the target path
        """

        # filter all unique dir paths
        unique_tgt_folders = paths_df["tgt"].unique()

        logger.debug("Unique target folders: %s", unique_tgt_folders)

        # create a folder for each unqiue dir path
        list(
            map(
                lambda tgt_folder: self.create_tgt_folder(str(tgt_folder)),
                unique_tgt_folders,
            )
        )

    def _append_src_base_path(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apends the base source paths to the proved source column

        Args:
            df (pd.dataframe): Contains the source paths

        Returns
           (pd.dataframe): Dataframe containg the source paths with the base path
        """
        df["src"] = df["src"].apply(lambda x: os.path.join(self.src_base_path, x))
        return df

    def copy_single_image(self, src_path: str, tgt_path: str):
        """
        Copy image from source path to target path.

        Args:
            src_path (str): Source path of the image.
            tgt_path (str): target path to copy the image.

        """
        try:
            shutil.copy(src_path, tgt_path)
            return None
        except Exception as e:
            logger.error("Error at copyng %s to %s: %s", src_path, tgt_path, e)
            raise

    def _check_input(self, paths: Union[pd.DataFrame, List[Tuple[str, str]]]):
        """Check if the input is a DataFrame or a list of tuples containing source and target paths.
        Turn

        Args:
            paths (Union[pd.DataFrame, List[Tuple[str, str]]]):
                DataFrame with 'src' and 'tgt' columns or list of tuples (src, tgt).

        Returns:
            (pd.Dataframe): A dataframe containing a src and tgt column
        """
        if isinstance(paths, pd.DataFrame):
            if not {"src", "tgt"}.issubset(paths.columns):
                logger.error(
                    "Dataframe needs to have 'src' and 'tgt' columns. Current columns: %s",
                    paths.columns,
                )
                raise ValueError(
                    f"Dataframe needs to have 'src' and 'tgt' columns.{paths.columns}"
                )
            paths_df = paths[["src", "tgt"]].copy()
        elif isinstance(paths, list):
            paths_df = pd.DataFrame(paths, columns=["src", "tgt"])
        else:
            raise TypeError("paths must be a DataFrame or a list of tuples.")

        logger.info("Number of images to copy: %s", len(paths_df))

        return paths_df

    def _iterate_over_paths(self, paths_df: pd.DataFrame):
        """Simple iteration over the src and targets paths to copythe images

        Args:
            paths_df (pd.DataFrame): Df with target and souce paths

        Raises:
            Value Error: If not all images could been copied
        """
        success_count = 0
        failure_count = 0

        for _, row in paths_df.iterrows():
            src_path = row["src"]
            tgt_path = row["tgt"]

            try:
                self.copy_single_image(src_path, tgt_path)
                success_count += 1
            except ValueError:
                failure_count += 1

        logger.info("%s of %s images copied successfully.", success_count, len(paths_df))
        if failure_count > 0:
            logger.error("Falied to copy %s images.", failure_count)
            raise ValueError

    def _parallel_image_copier(self, paths_df: pd.DataFrame, max_workers: int = None):
        """Copies the images with parallesization to make the copy faster.

        Args:
            paths_df (pd.DataFrame): dataframe containg the source and target path as columns
            max_worker (int): Number of workers to use for the copy job

        """

        success_count = 0
        failure_count = 0

        if max_workers is None:
            max_workers = min(32, 32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.copy_single_image, row["src"], row["tgt"])
                for idx, row in paths_df.iterrows()
            ]

            for future in as_completed(futures):
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failure_count += 1

    def copy_images(
        self,
        paths: Union[pd.DataFrame, List[Tuple[str, str]]],
        parallel: bool = False,
        max_workers: int = None,
    ):
        """
        Copy images based on a DataFrame or list of tuples containing source and target paths.

        Args:
            paths (pd.Dataframe | tuple): df or list of tuples containing source and target paths.
                                          Example of the DataFrame:

                                               | src            |   tgt                |
                                               |----------------|----------------------|
                                               | path/to/image1 | path/to/target_folder|

            parallel (bool): Flag to enable parallel processing.
            max_workers (int): Maximum number of workers for parallel processing.

        """
        paths_df = self._check_input(paths)

        self._prepare_target_folders(paths_df)

        if self.src_base_path is not None:
            paths_df = self._append_src_base_path(paths_df)

        if parallel:
            self._parallel_image_copier(paths_df, max_workers)
        else:
            self._iterate_over_paths(paths_df)

        logger.info("Images copied successfully.")

    def clean_folder(self, folder_path: str):
        """
        Cleans the folder at the specified path by deleting all diffrent contents

        Args:
            folder_path (str): Folder path to be cleaned.
        """
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                logger.info("Cleaned up: %s", folder_path)
            self.create_tgt_folder(folder_path)

            return None
        except Exception as e:
            logger.error("Error at removing elements inside %s: %s", folder_path, e)
            raise
