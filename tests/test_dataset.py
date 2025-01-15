import os
from datetime import datetime as dt
from datetime import timezone

import pytest
import pandas as pd
import numpy as np
from unittest.mock import mock_open, patch, call

from lit_ecology_classifier.archiv.create_overview_data_set import CreateOverviewDf


class TestCreateDataFrame:
    """Testing for the create_dataframe function"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test case"""
        with patch("os.path.exists") as mocked_exists:
            # Mock os.path.exists to return True for required paths
            mocked_exists.return_value = True
            self.create_dataset = CreateOverviewDf()

    # hash_image_sha256 -----------------------------------------------------------------------------------------------------

    @patch("builtins.open", side_effect=PermissionError)
    def test_hash_image_sha256_access_permision_error(self, mock_open):
        with pytest.raises(PermissionError):
            self.create_dataset._hash_image_sha256("invalid_path.jpeg")

    # find_images_in_folder -------------------------------------------------------------------------------------------------

    def test_find_images_in_folder(self):
        "test if the function returns the correct path and ignores non-image files"
        input_tuple = ("1", "fake_folder")
        folder_path = os.path.join("fake_folder", "test")
        files = ["image1.jpeg", "test.txt", "pickle.pkl"]
        expected_output = [("1", os.path.join(folder_path, "image1.jpeg"))]

        # patch os.walk to return the predefined folder path and files
        with patch("os.walk") as mock_walk:
            mock_walk.return_value = [
                (folder_path, [], files),
            ]

            assert (
                self.create_dataset._collect_image_paths_from_folder(input_tuple)
                == expected_output
            )

    # hotencoding_group_by -------------------------------------------------------------------------------------------------

    def test_hotencoding_group_by(self):
        """Test if the function returns the correct output"""

        # create mock data with class, dataset version, hash and date
        df = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image1.jpeg", "image2.jpeg", "image3.jpeg"],
                "class": ["A", "A", "C", "D"],
                "sha256": ["hash1", "hash1", "hash2", "hash3"],
                "data_set_version": ["1", "2", "2", "1"],
                "date": ["2021-01-01", "2021-01-01", "2021-01-03", "2021-01-03"],
            }
        )
        expected_output = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image2.jpeg", "image3.jpeg"],
                "class": ["A", "C", "D"],
                "sha256": ["hash1", "hash2", "hash3"],
                "date": ["2021-01-01", "2021-01-03", "2021-01-03"],
                "ZooLake_1": [True, False, True],
                "ZooLake_2": [True, True, False],
            }
        )

        pd.testing.assert_frame_equal(
            self.create_dataset._add_one_hot_encoded_versions_and_group_by(df),
            expected_output,
        ), "The output is not correct"

    # check_duplicates -------------------------------------------------------------------------------------------------------

    def test_duplicate_check_duplicate(self):
        """Test if the function raises a warning when there are duplicates in the dataframe and
        the correct number of duplicates is printed"""
        input_df = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image2.jpeg"],
                "class": ["A", "A"],
                "data_set_version": ["1", "1"],
                "sha256": ["hash1", "hash1"],
                "date": ["2021-01-01", "2021-01-01"],
            },
            index=[0, 1],
        )

        excepted_output = pd.DataFrame(
            {
                "sha256": ["hash1", "hash1"],
                "data_set_version": ["1", "1"],
                "count": [2,2],
                "diffrent_class": [False, False],
                "diffrent_image_name": [True, True],
                "image": ["image1.jpeg","image2.jpeg"],
                "class": ["A","A"]
            },
            index=[0, 1],
        )

        with patch("warnings.warn") as mock_warn:
            output_df = self.create_dataset.check_duplicates(input_df)
            mock_warn.assert_called_once_with(f"Duplicates found in the dataset: 2")
            pd.testing.assert_frame_equal(output_df, excepted_output), print(output_df.head())

    def test_duplicate_check_no_duplicate(self):
        """Test if the function does not raise a warning if there are no duplicates in the dataframe"""
        df = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image2.jpeg"],
                "class": ["A", "B"],
                "data_set_version": ["1", "1"],
                "sha256": ["hash1", "hash2"],
                "date": ["2021-01-01", "2021-01-01"],
            }
        )
        assert self.create_dataset.check_duplicates(df) == None

    def test_duplicate_check_different_version(self):
        """Test if the function does not raise a warning if there are similar hashes but different dataset versions"""
        df = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image1.jpeg"],
                "class": ["A", "A"],
                "data_set_version": ["1", "2"],
                "sha256": ["hash1", "hash1"],
                "date": ["2021-01-01", "2021-01-01"],
            }
        )

        assert self.create_dataset.check_duplicates(df) == None
