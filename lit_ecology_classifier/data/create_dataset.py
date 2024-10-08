from datetime import datetime as dt
import imagehash
import os
import hashlib
from PIL import Image
import pickle  


from tqdm import tqdm

import numpy as np
import pandas as pd

class CreateOverviewDf():
    def __init__(self, zoolake_versions_dic=None , hash_algorithm = None):
        '''Initialize the dataset creator with the directory strcutures to the diffrent dataset versions'''
        if zoolake_versions_dic is None:
            zoolake_versions_dic = {
                '1': os.path.join('..', 'data', 'raw', 'data', 'zooplankton_0p5x'),
                '2': os.path.join('..', 'data', 'raw', 'ZooLake2', 'ZooLake2.0')
            }

        if hash_algorithm is None:
            hash_algorithm = 'sha256'
        
        if hash_algorithm not in ['sha256', 'phash']:
            raise ValueError(f'Invalid hash algorithm: {hash_algorithm}. Choose between "sha256" and "phash"')	
        

        self.hash_algorithm = hash_algorithm
        self.dic_dataset = zoolake_versions_dic
        self.images_dict = {}
        self.overview_df = None
        self.duplicates_hashes = None

    
    def hash_image_sha256(self, image_path):
        '''Calculate the SHA256 hash of the binary image data
        
        Args:
            image_path (str): Path to the image file

        Returns:
            str: SHA256 hash of the

        Raises:
            Exception: If the image cannot be read and hashed
        '''
        try:
            
            with Image.open(image_path) as img:

                # check if image_path ends with .jpeg or .png
                if img.format not in ['JPEG', 'PNG']:
                    raise ValueError(f'Error hashing {image_path} with SHA256: Invalid file format {img.format}')
                
                # Read the binary data of the image
                img_data = img.tobytes()

                # Calculate the SHA256 hash of the image data
                return hashlib.sha256(img_data).hexdigest()
            
        except Exception as e:
            if isinstance(e, PermissionError):
                raise PermissionError(f'Error hashing {image_path} with SHA256: Permission denied')
            else:
                raise Exception(f'Error hashing {image_path} with SHA256: {e}')


    def hash_image_phash(self, image_path):
        '''Calculate the perceptual hash (pHash) of the image.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: pHash of the image
            
        Raises:
            Exception: If the image cannot be read and hashed
        '''
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))
        except Exception as e:
            print(f'Error hashing {image_path} with pHash: {e}')
            return None


    def get_timestamp_from_filename(self, image_path):
        '''Extract the timestamp from the image filename and convert it to a datetime object.
        
        Args:
            image_path (str): Path to the image file

        Returns:
            datetime: Timestamp extracted from the filename as a datetime object
            
        Raises:
            Exception: If the timestamp cannot be transformed to a datetime object
        '''
              
        try:
            image_name = os.path.basename(image_path)

            # Extract the timestamp part and keep only the first 10 characters (seconds)
            timestamp_str = image_name[15:25] 
            timestamp = int(timestamp_str)  # Convert to seconds
            return dt.fromtimestamp(timestamp)
        except Exception as e:
            raise ValueError(f'Error extracting and creating timestamp from {image_path}: {e}')


    def get_plankton_class(self, image_path, version):
        '''Extract the plankton class from the image path.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Plankton class extracted from the image path
        '''

        if version == 1:
            return os.path.basename(os.path.dirname(os.path.dirname(image_path)))

        else:      
            return os.path.basename(os.path.basename(os.path.dirname(image_path)))



    def process_image(self, image_path, version):
        '''Process a single image to compute the hashes and extract metadata.
        
        Args: 
            image_path (str): Path to the image file
            plankton_class (str): Class of the plankton in the image
            version (int): Version of the ZooLake dataset
            
        Returns:
            dict: Dictionary containing the image metadata and hashes
        
        Raises:
            Exception: If the image cannot be processed
        '''
        try:
            image_hash_sha256 = self.hash_image_sha256(image_path)
            image_date = self.get_timestamp_from_filename(image_path)
            plankton_class = self.get_plankton_class(image_path, version)
            if self.hash_algorithm == 'phash':
                image_hash_phash = self.hash_image_phash(image_path)
            

                return {
                        'image': os.path.basename(image_path),
                        'phash': image_hash_phash,
                        'class': plankton_class,
                        'data_set_version': int(version),
                        'date': image_date
                    }
            else:
                return {
                        'image': os.path.basename(image_path),
                        'sha256': image_hash_sha256,
                        'class': plankton_class,
                        'data_set_version': int(version),
                        'date': image_date
                    }

        except Exception as e:
            raise Exception(f'Error processing image {image_path}: {e}')


    def find_images_in_folder(self, folder_path):
        '''Find all images in a folder and its subfolders.
        
        Args:
            folder_path (str): Path to the folder
        
        Returns:
            list: List of image paths
        '''
        return [os.path.join(root, file)    # join the root path with the file name
                for root, _, files in os.walk(folder_path) # walk through the folder and subfoledrs (generates lists)
                for file in files if file.endswith('.jpeg')] # filter for files that end with .jpeg
    

    def map_image_list_and_processing(self, file_list, version):
        '''Applies the image processing function to a list of image paths 

        Args:
            file_list (list): List with the image paths
            version (int): Version of the ZooLake dataset

        Returns:
            None
        '''
        a = map(self.process_image, file_list, [version]*len(file_list))

        self.images_dict.update(a)      
        return None


    def genrate_image_list(self):

        for version, plankton_classes_overview_path in self.dic_dataset.items():
            # Get all images in the folder
            image_paths = self.find_images_in_folder(plankton_classes_overview_path)
            
            a = map(self.process_image, image_paths, [version]*len(image_paths))

            self.images_dict.extend(a)

        return self.images_dict


    def check_duplicates(self, df):
        '''Check for duplicates in the same dataset version
        
        Args:
            df (pd.DataFrame): DataFrame containing the image metadata and hashes
        
        Returns:
            None
        '''

        if self.hash_algorithm == 'sha256':

            # save the duplicates hash values in a list
            self.duplicates_hashes = df[df.duplicated(subset=['data_set_version',self.hash_algorithm], keep=False)][self.hash_algorithm].tolist()
            
            if len(self.duplicates_hashes) > 0:
                raise Warning(f'Duplicates found in the dataset: {len(self.duplicates_hashes)}')
            
            else:
                print('No duplicates found in the dataset')
                return None


    def hotencode_versions_and_group_by(self, df):
        '''One-hot encode the classes in the DataFrame and group by the hash value
        
        Args:
            df (pd.DataFrame): DataFrame containing the image metadata and hashes
        '''

        # One-hot encode the data_set_versions column
        df = pd.get_dummies(df, columns=['data_set_version'], drop_first=False , prefix='ZooLake')

        # group by hash value an keep the maximum value of each columns in the group and preserve the columns order
        df = df.groupby([self.hash_algorithm], as_index = False, sort = False ).max()
    
        return df


    def load_test_train_from_pickle(self, path_pickle_file ):
        '''Load the test and train sets from the pickle file
        
        Args:
            path_pickle_file (str): Path to the pickle file
        
        Returns:
            tuple: Tuple containing the training and test sets
        '''
        train_test_val = pd.read_pickle(path_pickle_file)

        return train_test_val


    def load_test_train_from_txt(self, datapath):
        filenames_train = np.loadtxt(datapath + '/zoolake_train_test_val_separated/train_filenames.txt', dtype=str)
        filenames_val = np.loadtxt(datapath + '/zoolake_train_test_val_separated/val_filenames.txt', dtype=str)
        filenames_test = np.loadtxt(datapath + '/zoolake_train_test_val_separated/test_filenames.txt', dtype=str)
        return [filenames_train, filenames_test, filenames_val]


    def update_each_test_train_split(self, train_test_val_array ):
        '''Split the DataFrame into training and test sets based on the pickle file 
        
        Args:
            df (pd.DataFrame): DataFrame containing the image metadata and hashes
        
        Returns:
            df: DataFrame containing the training set and test set as columns
        '''
        # Split the DataFrame into training and test sets

       
        splits = ['train', 'test', 'val']

        for versions,  in enumerate(train_test_val_array):
            for i,split in enumerate(train_test_val_array):
                train_test_val_array[i]
                self.update_df_with_split_image_names(df, train_test_val_array[i], splits[i])

        return None


    def update_df_with_split(self,df, lst, split_name):

        lst = [os.path.basename(file_path) for file_path in lst]
  
        df[split_name] = df['image'].isin(lst)

        return df

    
if __name__ == '__main__':
    print('Running the dataset creator')
    dataset_creator = CreateOverviewDf(zoolake_versions_dic = {
                '1': os.path.join('data', 'raw', 'data'),
                '2': os.path.join('data', 'raw', 'ZooLake2')
            })
    
    df = pd.DataFrame({
            'class': ['A', 'A', 'C', 'D'],
            'data_set_version': [1, 2, 2, 1],
            'sha256': ['hash1', 'hash1', 'hash2', 'hash3'],
            'date': ['2021-01-01', '2021-01-01', '2021-01-03', '2021-01-03']
        })
    
    print(dataset_creator.images_dict)

