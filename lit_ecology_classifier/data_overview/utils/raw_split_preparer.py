"""
Prepare the file paths for the train/test/val splits for each ZooLake dataset versions 
before further processing to ensure a faster user feedback of not existing files
containing the splits.
"""

import logging
import os
import warnings


class _RawSplitPathPreparer:
    """Prepare the file paths for the train/test/val splits for each ZooLake dataset versions.

    The RawSplitPathPreparer is used to prepare the file paths for the train/test/val 
    splits for each version of the ZooLake dataset.It checks if the split files exist
    and generates the paths to the train/test/val .txt files or the pickle file containing 
    the splits  to be used for further processing in a later step.

    Attributes:
        zoolake_version_paths (dict): Dictionary containing the paths to the different
                                        ZooLake dataset versions
        _split_file_paths (dict): Dictionary containing the paths to the train/test/val txt
                                     files or the pickle file
    """

    def __init__(self, zoolake_version_paths: dict):
        """Initialize the RawSplitLoader with the paths to the different ZooLake dataset versions.

        Args:
            zoolake_version_paths (dict): Dictionary containing the paths to the different
                                             ZooLake dataset versions.
        """
        self.zoolake_version_paths = zoolake_version_paths
        self._split_file_paths = {}

    def _prepare_split_paths_from_txt(self, datapath: str, version: str) -> dict:
        """Pepare and  if the txt files containing thesplits exist in the data folder for version 1.

        It assumes that the txt files are stored in the folder  'zoolake_train_test_val_separated'
        within the provided datapath. This only applies to version 1. After ensuring the existence 
        of the txt files,it generates the paths to the train/test/val .txt files 
        and stores them in a dictionary.

        Args:
            datapath (str): Path to the folder containg the data for ZooLake dataset
                             V1 as string
            version (str): Version of the ZooLake dataset as string to generate a key
                             for the path inside the path dictionary

        Returns:
           dict: Dictionary containing the data set version and subdictionory with the paths to the
                             train/test/val .txt filesas values and the split name as key.

        Raises:
            FileNotFoundError: If a txt files do not exist in the folder
        """
        # generate path to the folder containing the txt files
        path_txt_file_folder = os.path.join(
            datapath, "zoolake_train_test_val_separated"
        )

        # create dict to store the paths to the diffrent split .txt files
        self._split_file_paths[version] = {
            "train": os.path.join(path_txt_file_folder, "train_filenames.txt"),
            "test": os.path.join(path_txt_file_folder, "test_filenames.txt"),
            "val": os.path.join(path_txt_file_folder, "val_filenames.txt"),
        }

        #  if a txt file does not exist in the folder
        missing_files = [
            file
            for file in self._split_file_paths[version].values()
            if not os.path.exists(file)
        ]

        # raise a error if a file is missing
        if missing_files:
            raise FileNotFoundError(
                f"The following files are missing for version {version}: {', '.join(missing_files)}"
            )

        return self._split_file_paths

    def _prepare_split_paths_from_pickle(self, datapath: str, version: str) -> dict:
        """Search if a pickle file containing the splits exist in the folder

        Searches if they are any pickle files inside the datapath. The pickle file 
        is expected to contain the train/test/val splits for the given ZooLake dataset version.
        If the search does not return a result, it raises a warning, to inform the user that no 
        pickle file was found. It does nto raise an error, as a train/test/val split is not
          mandatory for each version of the dataset.

        When the search results withmultiple pickle files, it raises a valueerror. 
        Args:
            datapath (str): Path to the folder containg the data for the given ZooLake Version 
            version (str): Version of the ZooLake dataset as string to generate a key for 
                            the path inside the path dictionary

        Returns:
            dict: Dictionary containing the path to the pickle file for the train/test/val splits
                    Example: {"2": {"pickle": "/path/to/pickle/file.pickle"}}

        Raises:
            FileNotFoundError: If no pickle file is found in the folder
            ValueError: If multiple pickle files are found in the folder, as only one pickle 
                        file is expected
        """
        pickle_files = [
            file
            for file in os.listdir(datapath)
            if file.endswith(".pickle") or file.endswith(".pkl")
        ]

        #  if a pickle file exists
        if len(pickle_files) == 0:
            logging.warning(
                "No pickle file found for the dataset '%s' inside of %s",
                version,
                datapath,
            )
            warnings.warn(
                f"No pickle file found for the dataset '{version}' inside of {datapath}"
            )
            return None

        #  if there are multiple pickle files
        if len(pickle_files) > 1:
            raise ValueError(
                f"Multiple pickle files found in {datapath}. Please provide only one pickle file"
            )

        # Prepare file path for the pickle file
        path_pickle_file = os.path.join(datapath, pickle_files[0])
        self._split_file_paths[version] = {"pickle": path_pickle_file}

        return self._split_file_paths

    def prepare_split_paths(self) -> dict:
        """Prepare the file paths for the train/test/val splits for each ZooLake dataset version

        Prepares the file paths for different versions of the ZooLake dataset.
        - For version 1: it looks for train/test/val splits stored as .txt files.
        - For any other version: it assumes that the splits are stored in a pickle file.


        The preparation is unboudled from the loading of the splits, to be able to  the existence of
        missing files at an early stage.

        Args:
            None

        Returns:
            dict: Dictionary containing the paths to the train/test/val txt files or the pickle file

        Raises:
            Warning: If a new version is found, it assumes that the split is stored in a pickle file
        """

        self._split_file_paths = {}

        for version, datapath in self.zoolake_version_paths.items():
            if version == "1":
                # Prepare file paths for the txt files (train/test/val)
                self._prepare_split_paths_from_txt(datapath, version)


            else:
                # raise warning if the version is not 2, since currently only version 2 is released
                if version != "2":
                    warnings.warn(
                        "New version, assuming a pickle file for split in the folder"
                    )

                self._prepare_split_paths_from_pickle(datapath, version)

        return self._split_file_paths
