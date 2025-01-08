"""
Provides the main functionalities to copy images from a source directory to a destination directory.
Besides the copy functionality, it provides functionality to create and clean the target folder if needed.
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
    Base class for copying images from a source directory to a target directory with the `copy_iamge` method. 
    Before copying the images, it ensures that thearget folders exist and creates them if they do not exist. 
    The target folderS can also be e,ptiedbefore copying the images with the method.

    The images to copy should be be provided to the `copy_iamge` method as a DataFrame or a list of tuples containing the source and 
    target paths. The dataframe should have the columns 'src' and 'tgt' and the list of tuples should have the source path as the first
    element and the target path as the second element to ensure the correct transformation to a DataFrame. 

    Attributes:
        src_base_path (str): Source base path to the folder containing the class folder and images.
                                Should be used if only the class folder is provided in the source path.
        tgt_base_path (str): Target base path to the folder containing the class folder and images.

    Examples:

        ```python

        import pandas as pd

        from lit_ecology_classifier.helpers.base_copier import BaseImageCopier

        # example of a DataFrame
        paths = pd.DataFrame(
            { path/to/image1, path/to/target_folder},
            columns=["src", "tgt"]
        )

        # example of a list of tuples
        paths = [ (path/to/image1, path/to/target_folder)]

        copier = BaseImageCopier(src_base_path="path/to/source", tgt_base_path="path/to/target")
        copier.copy_images(paths)
        ```
    """

    def __init__(self, src_base_path: str = None, tgt_base_path: str = None):
        """ Initialize the BaseImageCopier with the base source and target paths.

        Args:
            src_base_path: Optional base path to the folder containing the
                                 class folders with images.
            tgt_base_path: Optional base path to the folder where the images should be copied.
        """
        self.src_base_path = src_base_path
        self.tgt_base_path = tgt_base_path

    def create_tgt_folder(self, folder_path: str):
        """Creates a folder at the specified target path after checking if it exists.

        The creation of the folder is done with the exist_ok=True parameter to avoid
        errors if the folder already exists.

        Args:
            folder_path (str): Path to the folder to be created.

        Raises:
            Exception: If an error occurs while creating the folder.
        """
        try:
            os.makedirs(folder_path, exist_ok=True)
            return None
        except Exception as e:
            logger.error("Error creating the folder %s: %s", folder_path, e)
            raise

    def _prepare_target_folders(self, paths_df: pd.DataFrame):
        """Prepare the target folders for the images to be copied.

        Finds all unique target folders in the provided DataFrame and creates them if they do not exist.

        Args:
            paths_df: A Dataframe containing the target path
        """

        # filter all unique target folders
        unique_tgt_folders = paths_df["tgt"].unique()

        logger.debug("Unique target folders: %s", unique_tgt_folders)

        # create a folder for each unqiue directory found in the target column
        list(
            map(
                lambda tgt_folder: self.create_tgt_folder(str(tgt_folder)),
                unique_tgt_folders,
            )
        )

    def _append_src_base_path(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apends the base source paths to the provided source column

        Args:
            df: A dataframe containing the source paths in the 'src' column

        Returns
           A Dataframe containg the source paths.
        """
        df["src"] = df["src"].apply(lambda x: os.path.join(self.src_base_path, x))
        return df

    def copy_single_image(self, src_path: str, tgt_path: str):
        """ Copy a single image from the source path to the target path.

        Args:
            src_path: Complete path to the image to be copied.
            tgt_path: Target path to copy the image. 

        """
        try:
            shutil.copy(src_path, tgt_path)
            return None
        except Exception as e:
            logger.error("Error at copyng %s to %s: %s", src_path, tgt_path, e)
            raise ValueError

    def _check_input(self, paths: Union[pd.DataFrame, List[Tuple[str, str]]]) -> pd.DataFrame:
        """Check if the input is a DataFrame or a list of tuples containing source and target paths.
        

        Args:
            paths:
                A DataFrame with 'src' and 'tgt' columns or list of tuples (src, tgt).

        Returns:
            A dataframe containing a src and tgt column
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
        """Simple iteration over the src and targets paths to copy the images

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
            paths_df: dataframe containg the source and target path as columns
            max_worker: Maximal number of workers for the ThreadPoolExecutor

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

        logger.info("%s of %s images copied successfully.", success_count, len(paths_df))

    def execute_copieng_of_images(
        self,
        paths: Union[pd.DataFrame, List[Tuple[str, str]]],
        parallel: bool = False,
        max_workers: int = None,
    ):
        """
        Copy images based on a DataFrame or list of tuples containing source and target paths.

        Args:
            paths: df or list of tuples containing source and target paths.
                                          Example of the DataFrame:

                                               | src            |   tgt                |
                                               |----------------|----------------------|
                                               | path/to/image1 | path/to/target_folder|

            parallel: Flag to enable parallel processing.
            max_workers: Maximum number of workers for parallel processing.

        """
        paths_df = self._check_input(paths)

        self._prepare_target_folders(paths_df)

        if self.src_base_path is not None:
            paths_df = self._append_src_base_path(paths_df)

        if parallel:
            self._parallel_image_copier(paths_df, max_workers)
        else:
            self._iterate_over_paths(paths_df)

    def clean_folder(self, folder_path: str):
        """
        Cleans the folder at the specified path by deleting all diffrent contents

        Args:
            folder_path: Folder path to be cleaned.
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
