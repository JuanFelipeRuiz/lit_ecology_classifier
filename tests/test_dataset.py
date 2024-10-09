import os
from datetime import datetime as dt
from datetime import timezone

import pytest
import pandas as pd
from unittest.mock import mock_open, patch

from lit_ecology_classifier.data.create_dataset import CreateOverviewDf

class TestCreateDataFrame:
    '''Testing for the create_dataframe function'''

    @pytest.fixture(autouse=True)
    def setup_method(self):
        '''Setup for each test case'''
        self.create_dataset = CreateOverviewDf()

    # get_timestamp_from_filename -------------------------------------------------------------------------------------------

    def test_get_timestamp_from_filename(self):
        file_name = 'SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg'
        expected_timestamp = dt.strptime('2019-10-08 14:02:52', '%Y-%m-%d %H:%M:%S')
        assert self.create_dataset.get_timestamp_from_filename(file_name) ==  expected_timestamp.replace(tzinfo=timezone.utc)


    def test_get_timestamp_from_filename_wrong_timestamp_format(self):
        file_name = 'SPC-EAWAG-0P5X-1A7054337290115-3725350526242-001629-055-1224-2176-84-64'
        with pytest.raises(ValueError):
            self.create_dataset.get_timestamp_from_filename(file_name)

    # get_plankton_class_datalake -------------------------------------------------------------------------------------------

    def test_get_plankton_class_datalake_v1(self):
        image_path = 'class/iamge.jpeg'
        expected_class = 'class'
        assert self.create_dataset.get_plankton_class(image_path,2) == expected_class

    def test_get_plankton_class_datalake_v2(self):
        image_path = 'class/training/iamge.jpeg'
        expected_class = 'class'
        assert self.create_dataset.get_plankton_class(image_path,1) == expected_class

    # hash_image_sha256 -----------------------------------------------------------------------------------------------------

    def test_hash_image_sha256_access_permision_error(self):
        with patch('builtins.open', side_effect=PermissionError):
            with pytest.raises(PermissionError):
                self.create_dataset.hash_image_sha256('invalid_path.jpeg')

    # process_image ---------------------------------------------------------------------------------------------------------

    def test_process_image_correct_output(self):
        image_name = 'SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg'
        image_path = os.path.join('fake_class','test', image_name)
        expected_output = {
            'image': image_name,
            'sha256': 'fake_sha256',
            'class': 'fake_class',
            'data_set_version': 1,
            'date': 'fake_timestamp',
        }

        # Mock the get_timestamp_from_filename method to return 'timestamp'
        with patch.object(self.create_dataset, 'get_timestamp_from_filename', return_value='fake_timestamp'):
            with patch.object(self.create_dataset, 'hash_image_sha256', return_value='fake_sha256'):
                with patch.object(self.create_dataset, 'get_plankton_class', return_value='fake_class'):
                    assert self.create_dataset.process_image(image_path, version = 1) == expected_output

    # find_images_in_folder -------------------------------------------------------------------------------------------------

    def test_find_images_in_folder(self):
        'test if the function returns the correct path and ignores non-image files'
        folder_path = os.path.join('fake_folder', 'test')
        files = ['image1.jpeg', 'image2.jpeg', 'image3.jpeg', 'test.txt', 'pickle.pkl']
        expected_output = [os.path.join(folder_path, image) for image in ['image1.jpeg', 'image2.jpeg', 'image3.jpeg']]

        # patch os.walk to return the expected output
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [
                (folder_path, [], files),
            ]

            assert self.create_dataset.find_images_in_folder(folder_path) == expected_output

    # hotencoding_group_by -------------------------------------------------------------------------------------------------

    def test_hotencoding_group_by(self):
        '''Test if the function returns the correct output'''

        # create mock data with class, dataset version, hash and date
        df = pd.DataFrame({
            'class': ['A', 'A', 'C', 'D'],
            'sha256': ['hash1', 'hash1', 'hash2', 'hash3'],
            'data_set_version': [1, 2, 2, 1],
            'date': ['2021-01-01', '2021-01-01', '2021-01-03', '2021-01-03']
        })
        expected_output = pd.DataFrame({
            'sha256': ['hash1', 'hash2', 'hash3'],
            'class': ['A', 'C', 'D'],
            'date': ['2021-01-01',  '2021-01-03', '2021-01-03'],
            'ZooLake_1': [True, False, True],
            'ZooLake_2': [True, True, False]
        }) 

        pd.testing.assert_frame_equal(self.create_dataset.hotencode_versions_and_group_by(df), expected_output), 'The output is not correct'


    # check_duplicates -------------------------------------------------------------------------------------------------------

    def test_duplcicate_check_duplicate(self):
        ''' Test if the function raises a warning when there are duplicates in the dataframe'''
        df = pd.DataFrame({
            'class': ['A', 'A'],
            'data_set_version': [1, 1],
            'sha256': ['hash1', 'hash1'],
            'date': ['2021-01-01', '2021-01-01']
        })

        with pytest.raises(Warning):
            self.create_dataset.check_duplicates(df)

    def test_duplcicate_check_no_duplicate(self):
        ''' Test if the function do not raise a warning, if they are no duplicates in the dataframe'''
        df = pd.DataFrame({
            'class': ['A', 'B'],
            'data_set_version': [1, 1],
            'sha256': ['hash1', 'hash2'],
            'date': ['2021-01-01', '2021-01-01']
        })
        assert self.create_dataset.check_duplicates(df) == None


    def test_duplcicate_check_diffrent_version(self):
        ''' Test if the function do not raise a warning, if they are similar hashesh but diffrent dataset versions'''
        df = pd.DataFrame({
            'class': ['A', 'A'],
            'data_set_version': [1, 2],
            'sha256': ['hash1', 'hash1'],
            'date': ['2021-01-01', '2021-01-01']
        })

        assert self.create_dataset.check_duplicates(df) == None

    def test_update_df_with_split_image_names(self):
        input_df = pd.DataFrame({
            'image': ['SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-76.jpeg', 'SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-77.jpeg'],
            'class': ['A', 'B'],
            'data_set_version': [1, 1],
            'sha256': ['hash1', 'hash2'],
            'date': ['2021-01-01', '2021-01-01']
        })

        train_lst = ['/home/EAWAG/chenchen/data/Zooplankton/train_data/training_zooplankton_new_220823///cyclops/SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-76.jpeg']

        expected_output = pd.DataFrame({
            'image': ['SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-76.jpeg', 'SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-77.jpeg'],
            'class': ['A', 'B'],
            'data_set_version': [1, 1],
            'sha256': ['hash1', 'hash2'],
            'date': ['2021-01-01', '2021-01-01'],
            'train_v1': [True, False]
        })

        pd.testing.assert_frame_equal(self.create_dataset.update_df_with_split_image_names(input_df, train_lst , 'train_v1'), expected_output)
