from datetime import datetime as dt
from datetime import timezone
import os
import hashlib
from PIL import Image
import pickle
import warnings

import imagehash
import numpy as np
import pandas as pd


class CreateOverviewDf:
    """ Generate an overview DataFrame based on the different ZooLake dataset versions, including images and metadata.

    Each image in the dataset is processed individually to extract metadata from its file name and folder path. The SHA256 hashing algorithm 
    is used to calculate a unique hash value for each image, which enables identical duplicate detection by identifying images with the same hash value. 
    The hash value also facilitates merging images from different dataset versions into a single DataFrame, with a one-hot encoded column indicating in 
    which data set versions the image occurs.

    Additional columns can be generated to indicate in which split (train, test, or validation) the image appears, based on a pickle files. For version 1,
    the split information is stored in separate .txt files and is also implemented in this class. The process assumes that the images are stored externally 
    with the original structure of the corresponding ZooLake dataset version.

    Attributes:
        zoolake_version_paths (dict): Maps dataset versions to their corresponding file paths.
        split_file_paths  (dict) = Maps dataset versions to their corresponding file paths for the train/test/validation splits. 
        hash_algorithm (str): Specifies the hashing algorithm to use, either "sha256" or "phash".
        images_metadata (dict): Stores the image metadata with corresponding hash values.
        overview_df (pd.DataFrame): DataFrame containing image metadata and hash values.
        overview_with_splits_df (pd.DataFrame): DataFrame containing image metadata, hashvalue and columns indicating which split (train/test/validation) the image belongs to.
        duplicates_df (pd.DataFrame):  DataFrame listing duplicate images in the dataset based on hash values.
    """

    def __init__(self, zoolake_version_paths = None, path_for_splits = None, hash_algorithm=None):
        """Initialize the overview DataFrame creator based on the dataset versions and hashing algorithm
        
        Args:
            zoolake_version_paths (dict): Maps dataset the versions to their corresponding file paths.
            hash_algorithm (str): Specifies the hashing algorithm to use, either "sha256" or "phash."
        
        """

        if zoolake_version_paths is None:
            zoolake_version_paths = {
                "1": os.path.join("..", "data", "raw", "data", "zooplankton_0p5x"),
                "2": os.path.join("..", "data", "raw", "ZooLake2", "ZooLake2.0"),
            }

        if hash_algorithm is None:
            hash_algorithm = "sha256"

        if hash_algorithm not in ["sha256", "phash"]:
            raise ValueError(
                f'Invalid hash algorithm: {hash_algorithm}. Choose between "sha256" and "phash"'
            )

        self.hash_algorithm = hash_algorithm
        self.zoolake_version_paths = zoolake_version_paths
        self.split_file_paths = None
        self.images_dict = []
        self.overview_df = None
        self.duplicates_df = None
        



    def hash_image_sha256(self, image_path : str) -> str:
        """Calculate the hash from the binary data of the image using the SHA256 algorithm.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: hash value of the image

        Raises:
            PermissionError: If the image cannot be read due to permission issues
            Exception: If the image cannot be hashed due to other issues
        """
        try:

            with Image.open(image_path) as img:

                # check if image_path ends with .jpeg or .png
                if img.format not in ["JPEG", "PNG"]:
                    raise ValueError(
                        f"Error hashing {image_path} with SHA256: Invalid file format {img.format}"
                    )

                # Read the binary data of the image
                img_data = img.tobytes()

                # Calculate the SHA256 hash based on the binary data of the image
                return hashlib.sha256(img_data).hexdigest()

        except Exception as e:
            if isinstance(e, PermissionError):
                raise PermissionError(
                    f"Error hashing {image_path} with SHA256: Permission denied"
                )
            else:
                raise Exception(f"Error hashing {image_path} with SHA256: {e}")

    def hash_image_phash(self, image_path : str) -> str:
        """Calculate the hash from the binary data of the image using the SHA256 algorithm.

        It is not recommended to use pHash for duplicate detection, as it is not unique for similar images and raises false positives.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: pHash value of the image

        Raises:
            PermissionError: If the image cannot be read due to permission issues
            Exception: If the image cannot be hashed due to other issues
        """
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))
        except Exception as e:
            if isinstance(e, PermissionError):
                raise PermissionError(
                    f"Error hashing {image_path} with phash: Permission denied"
                )
            else:
                raise Exception(f"Error hashing {image_path} with phash: {e}")

    def extract_timestamp_from_filename(self, image_path : str) -> dt:
        """Extract the timestamp from the image filename and convert it to a datetime object in UTC timezone.

        Args:
            image_path (str): Path to the image file

        Returns:
            datetime: Timestamp extracted from the filename as a datetime object with UTC as timezone

        Raises:
            ValueError: If the extracted value cannot be converted to a timestamp
        """

        try:

            # Extract the image name from the path
            image_name = os.path.basename(image_path)

            # Extract the timestamp part and keep only the first 10 characters (ignoring mili seconds)
            timestamp_str = image_name[15:25]

            # return the timestamp as a datetime object with UTC as timezone
            return dt.fromtimestamp(int(timestamp_str), tz=timezone.utc)

        except Exception as e:
            raise ValueError(
                f"Error extracting and creating timestamp from {image_path}: {e}"
            )

    def extract_plankton_class(self, image_path : str, version : str) -> str:
        """Extract  plankton class from the image path (Parent folder name).

        Handling of dataset version 1 and 2 is different, since dataset version 1 has an 
        additional parent folder 'training' between the image file and the plankton class folder. 

        Args:
            image_path (str): Path to the image file
            version (str): Version of the ZooLake dataset

        Returns:
            (str): A string containing the plankton class name
        """

        if version == '1':

            # Get the second parent / grandparent directory for ZooLake version 1
            return os.path.basename(os.path.dirname(os.path.dirname(image_path)))

        else:

            # get the parent directory for each other ZooLake version
            return os.path.basename(os.path.basename(os.path.dirname(image_path)))

    def process_image(self, image_path : str, version : str) -> dict:
        """Process a single image to calculate the hash and extract metadata from the filename and path.

        Args:
            image_path (str): Path to the image file
            version (str): Version of the ZooLake dataset

        Returns:
            dict: Dictionary containing the image metadata and hashes

        Raises:
            Exception: If the image cannot be processed
        """
        try:

            image_date = self.extract_timestamp_from_filename(image_path)
            plankton_class = self.extract_plankton_class(image_path, version)

            if self.hash_algorithm == "phash":
                image_hash_phash = self.hash_image_phash(image_path)

                return {"image" : os.path.basename(image_path),
                    
                    "phash": image_hash_phash,
                    "class": plankton_class,
                    "data_set_version": version,
                    "date": image_date,
                }
            
            else:
                image_hash_sha256 = self.hash_image_sha256(image_path)
                return{
                    "image": os.path.basename(image_path),
                    "sha256": image_hash_sha256,
                    "class": plankton_class,
                    "data_set_version": version,
                    "date": image_date,
                }
            
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {e}")

    def find_images_in_folder(self, folder_path):
        """Find all images with extension .jpeg inside a folder and its subfolders recursively.

        Args:
            folder_path (str): Path to the folder

        Returns:
            list: List of image paths
        """
        return [
             # join the root path with the file name
            os.path.join(root, file) 
            # walk through the folder and subfoledrs (generates lists of filespath and filenames)
            for root, _, files in os.walk(folder_path)  
            # loop through the files in the folder
            for file in files
            # filter for files that end with .jpeg
            if file.endswith(".jpeg")
        ]  

    def map_image_list_and_processing(self, file_list, version):
        """Applies the image processing function to a list of image paths

        Args:
            file_list (list): List with the image paths
            version (int): Version of the ZooLake dataset

        Returns:
            self.images_dict (list): List of dictionaries containing the image metadata and hashes
        """
        processed_images = map(self.process_image, file_list, [version] * len(file_list))

        return self.images_dict.extend(processed_images)

    def genrate_image_list(self):

        for version, plankton_classes_overview_path in self.zoolake_version_paths.items():

            # Get all images in the folder of the plankton class 
            image_paths = self.find_images_in_folder(plankton_classes_overview_path)

            self.images_dict = self.map_image_list_and_processing(image_paths, version)

        return self.images_dict
    

    def check_duplicates(self,df):
        """Check for duplicates in the same dataset version

        The duplicates are identified by comparing the hash values of the images in each dataset.
        When a duplicate is found, a DataFrame is created with further information, if the duplicate images ave the same class or image name.

        Args:
            df (pd.DataFrame): DataFrame containing the image metadata and hashes

        Returns:
            pd.DataFrame: DataFrame containing the duplicates in the dataset based on hash values
        """

        if self.hash_algorithm == "sha256":

            # save the duplicates hash values in a list
            duplicates = df[df.duplicated(subset=["sha256", 'data_set_version'], keep=False)].copy()

            if not duplicates.empty:

                # Group by hash_col and DataSetVersion
                group_counts = (
                    duplicates.groupby(["sha256", 'data_set_version'])
                    .agg(
                        # Count the number of duplicates
                        count=('class', 'size'),  

                        # Check if the class and image name are the same for all duplicates
                        diffrent_class=('class', lambda x: x.nunique() != 1), 
                        diffrent_image_name=('image', lambda x: x.nunique() != 1)
                    )
                    .reset_index()
                )
                
                group_counts['count'] = group_counts['count'].astype(int)

                self.duplicates_df =  group_counts[group_counts['count']>0]


                warnings.warn(
                    f"Duplicates found in the dataset: {duplicates.shape[0]}"
                )

                return self.duplicates_df

            else:
                print("No duplicates found in the dataset")
                return None
            
        else:
            warnings.warn("Duplicates check is only available for sha256 hash algorithm since \
                           the phash is not unique for similar images and raises false positives")

    def hotencode_versions_and_group_by(self, df):
        """One-hot encode the classes in the DataFrame and group by the hash value

        Args:
            df (pd.DataFrame): DataFrame containing the image metadata and hashes
        """

        # One-hot encode the data_set_versions column
        df = pd.get_dummies(
            df, columns=["data_set_version"], drop_first=False, prefix="ZooLake"
        )

        # group by hash value an keep the maximum value of each columns in the group and preserve the columns order
        df = df.groupby([self.hash_algorithm ], as_index=False, sort=False).max()

        return df
    

    def prepare_file_paths_txt(self, datapath, version):
        """ Prepare and check if the  txt files containing the train/test/val splits exist in the folder

        Args:
            datapath (str): Path to the folder containg the data for the ZooLake dataset V1
            version (str): Version of the ZooLake dataset

        Returns:
            dict: Dictionary containing the paths to the train/test/val txt files
        
        Raises:
            FileNotFoundError: If a txt files do not exist in the folder

        """
        # generate path to the folder containing the txt files
        path_txt_file_folder = os.path.join(datapath, "zoolake_train_test_val_separated")

        # create dict to store the paths to the diffrent split .txt files 
        self.split_file_paths[version] = {
                    'train': os.path.join(path_txt_file_folder, "train_filenames.txt"),
                    'test': os.path.join(path_txt_file_folder, "test_filenames.txt"),
                    'val': os.path.join(path_txt_file_folder, "val_filenames.txt")
                }

        # check if all the files exist
        if not all([os.path.exists(file) for file in self.split_file_paths[version].values()]):
            raise FileNotFoundError(f"Files for version {version} do not exist")
        
        return self.split_file_paths
        

    def prepare_file_paths_pickle(self, datapath, version):
        """ Prepare and check if the pickle file containing the train/test/val splits exist in the folder
        
        Args:
            datapath (str): Path to the folder containg the data for the given ZooLake Version
            version (str): Version of the ZooLake dataset

        Returns:
            dict: Dictionary containing the path to the pickle file

        Raises:
            FileNotFoundError: If the pickle file does not exist
            ValueError: If multiple pickle files are found in the folder
        """
        
        # search for pickle file in the folder
        pickle_files = [file for file in os.listdir(datapath) if file.endswith(".pickle")]

        # check if the pickle file exists
        if len(pickle_files) == 0:
            raise FileNotFoundError(f"No pickle file found in {datapath}")
        
        # check if there are multiple pickle files        
        if len(pickle_files) > 1:
            raise ValueError(f"Multiple pickle files found in {datapath}. Please provide only one pickle file")
                
        # Prepare file path for the pickle file
        path_pickle_file = os.path.join(datapath, pickle_files[0])
        self.split_file_paths[version] = {
                    'pickle': path_pickle_file
                }
        return self.split_file_paths

    
    def prepare_file_paths_splits(self):
        """ Prepare the file paths for the train/test/val splits for the different ZooLake dataset versions 

        Args:
            None

        Returns:
            dict: Dictionary containing the paths to the train/test/val txt files or the pickle file

        Raises:
            Warning: If the version is not 2, since currently only version 2 is available and it is assumed 
                     that the split is stored in a pickle file
        """
        
        self.split_file_paths = {}

        for version, datapath in self.zoolake_version_paths.items():
            if version == '1':
                # Prepare file paths for the txt files (train/test/val)
                self.prepare_file_paths_txt(datapath, version)
            
            else:
                # raise warning if the version is not 2, since currently only version 2 is released
                if version != '2':
                    warnings.warn("New version, assuming a pickle file for split in the folder")

                # Prepare file paths for the pickle file
                self.prepare_file_paths_pickle(datapath, version)
        
        return self.split_file_paths


    def load_split_overview_from_pickle(self, version):
        """ Load the train/val and test image splits from a pickle file into a dictionary

        It assumes that the pickle file contains a DataFrame with the image paths for the train, test, and validation splits.

        Args:
            path_pickle_file (str): Path to the pickle file

        Returns:
            Dictionary containing the imagepath to train/test and validation split

        Raises:
            ValueError: If an error occurs while unpickling the file
            Exception: If an unexpected error occurs while loading the pickle file
        """

        splits = ["train", "test", "val"]

        # Get the path to the pickle file from the prepared paths dictionary
        path_pickle_file = self.split_file_paths[version]['pickle']
        
        try:
            # read the pickle file
            train_test_val_splits = pd.read_pickle(path_pickle_file)

            # create a dictionary with the splits and the corresponding image paths
            return {split: train_test_val_splits[i] for i, split in enumerate(splits)}
        
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Error unpickling file {path_pickle_file}: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while loading the pickle file: {e}")

        

    def load_split_overview_from_txt(self, version):
        """ Load the train/val and test image splits from diffrent .txt files

        Args:
            path_txt_file_folder (str): Path to the folder containing the .txt files

        Returns:
            Dictionary containing the imagepath to train/test and validation split

        """

        splits = ["train", "test", "val"]

        # Get the paths from the prepared paths dictionary
        path_txt_files = self.split_file_paths[version]

        return {split: np.loadtxt(path_txt_files[split], dtype=str) for split in splits}
    
    def get_data_splits_by_version(self,df):
        """ Get the data splits for the different versions of the ZooLake dataset

        Args:
            df (pd.DataFrame): DataFrame containing at least the image file names
        
        Returns:
            pd.DataFrame: DataFrame containing the training set and test set as columns
        
        """
        # Loop through the different versions of the ZooLake dataset
        for version in self.zoolake_version_paths.keys():
            
            if version == '1':
                # if the version is 1, load the split from the txt files
                images_paths_split = self.load_split_overview_from_txt(version)

            else:

                # load the split from the pickle file
                images_paths_split = self.load_split_overview_from_pickle(version)
                
            df = self.update_dataframe_with_splits(images_paths_split, version, df)

        return df

    def update_dataframe_with_splits(self, images_paths_split, version, df):
        """Add columns to the DataFrame indicating the correspondity to the training,test or validation 

        Args:
            df (pd.DataFrame): DataFrame containing the image metadata and hashes
            images_paths_split (dict): Dictionary containing the image paths for the training, test, and validation splits
            version (str): Version of the ZooLake dataset

        Returns:
            df: DataFrame containing the training set and test set as columns
        """
       
        for split_name, image_paths in images_paths_split.items():
                
                #generate the column name based on the split name and the version
                column_name = f"{split_name}_v{version}"

                df = self.add_split_group_column(df = df, image_paths = image_paths, split_name = column_name)
                
        return df

    def add_split_group_column(self, df, image_paths, split_name):
        """Add a column to the DataFrame indicating whether an image is in the split

        Args:
            df (pd.DataFrame): DataFrame containing the image names
            image_paths (list): List of image paths for the split
            split_name (str): Name of the split column to add

        Returns:
            pd.DataFrame: DataFrame containing a new column indicating the split 
        """
        
        # extract the image names from the image paths
        lst = [os.path.basename(image_path) for image_path in  image_paths]

        # add a column to the DataFrame indicating whether the image is in the split or not
        df[split_name] = df["image"].isin(lst)

        return df
    
    def get_raw_df(self):
        if self.images_dict is None:
            self.genrate_image_list()
        return pd.DataFrame(self.images_dict)
    
    def get_overview_df(self):
        if self.overview_df is None:
            df = self.get_raw_df()
            self.check_duplicates(df)
            self.overview_df =self.hotencode_versions_and_group_by(df)
        return self.overview_df
    
    
    def get_overview_with_splits_df(self):
        if self.overview_with_splits_df is None:
            self.overview_with_splits_df = self.main()
        return self.overview_with_splits_df

    
    def main(self , load_new = False):
        """Main function to create the overview DataFrame

        Returns:
            pd.DataFrame: DataFrame containing the image metadata and hashes
        """

        if self.split_file_paths is None or load_new:
            self.prepare_file_paths_splits()

        if self.images_dict is None or load_new:
            self.get_raw_df()

        if self.overview_df is None or load_new:
            self.get_overview_df()

        if self.overview_with_splits_df is None or load_new:
            self.overview_with_splits_df = self.get_data_splits_by_version(self.overview_df)

        return self.overview_with_splits_df
       


if __name__ == "__main__":
    print("Running the dataset creator")
    dataset_creator = CreateOverviewDf(
        zoolake_version_paths={
            "1": os.path.join("data", "raw", "data"),
            "2": os.path.join("data", "raw", "ZooLake2"),
        }
    )

    print(dataset_creator.main())   