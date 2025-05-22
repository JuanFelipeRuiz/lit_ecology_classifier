import os
from unittest.mock import patch
import pytest
import pandas as pd
from lit_ecology_classifier.data_overview.zoolake_overview import ZooLakeOverviewCreator


class TestCreateOverviewDf:
    """Test suite for the CreateOverviewDf class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test case."""
        with patch("os.path.exists", return_value=True):
            self.create_dataset = ZooLakeOverviewCreator()

    # Test find_images_in_folder ------------------------------------------------------------------------------------

    def test_find_images_in_folder(self):
        """Test if the function returns the correct path and ignores non-image files."""

        input_tuple = ("1", "fake_folder")

        folder_path = os.path.join("fake_folder", "test")

        files = ["image1.jpeg", "test.txt", "pickle.pkl"]
        expected_output = [os.path.join(folder_path, "image1.jpeg")]

        with patch("os.walk", return_value=[(folder_path, [], files)]):
            assert (
                self.create_dataset._collect_image_paths_from_folder(input_tuple)
                == expected_output
            )

    # Test hotencoding_group_by -------------------------------------------------------------------------------------

    def test_hotencoding_group_by(self):
        """Test if the function returns the correct output."""
        df = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image1.jpeg", "image2.jpeg", "image3.jpeg"],
                "image_path": ["version1/image1.jpeg", "version2/image1.jpeg", "version2/image2.jpeg", "version1/image3.jpeg"],
                "class": ["A", "A", "C", "D"],
                "hash": ["hash1", "hash1", "hash2", "hash3"],
                "dataset": ["1", "2", "2", "1"],
                "date": ["2021-01-01", "2021-01-01", "2021-01-03", "2021-01-03"],
            }
        )

        # Output frame should take the image path from version 1 since it has more entries than version 2
        expected_output = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image2.jpeg", "image3.jpeg"],
                "image_path": ["version1/image1.jpeg", "version2/image2.jpeg", "version1/image3.jpeg"],
                "class": ["A", "C", "D"],
                "hash": ["hash1", "hash2", "hash3"],
                "date": ["2021-01-01", "2021-01-03", "2021-01-03"],
                "dataset_1": [True, False, True],
                "dataset_2": [True, True, False],
            }
        )

        pd.testing.assert_frame_equal(
            self.create_dataset._add_one_hot_encoded_versions(df),
            expected_output, check_exact=False, check_like=True,
        ) 


    def test_hotencoding_group_by_with_unlabelled(self):
        """Test if the function returns the correct output with unlabelled data."""
        df = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image1.jpeg", "image2.jpeg", "image3.jpeg"],
                "image_path": ["version1/image1.jpeg", "unlabelled/image1.jpeg", "unlabelled/image2.jpeg", "unlabelled/image3.jpeg"],
                "class": ["A", "A", "C", "D"],
                "hash": ["hash1", "hash1", "hash2", "hash3"],
                "dataset": ["1", "unlabelled", "unlabelled", "unlabelled"],
            }
        )

        # Output frame should take the image path from version 1 since it has more entries than version 2
        expected_output = pd.DataFrame(
            {
                "image": ["image1.jpeg", "image2.jpeg", "image3.jpeg"],
                "image_path": ["version1/image1.jpeg", "unlabelled/image2.jpeg", "unlabelled/image3.jpeg"],
                "class": ["A", "C", "D"],
                "hash": ["hash1", "hash2", "hash3"],
                "dataset_1": [True, False, False],
                "dataset_unlabelled": [True, True, True],
            }
        )

        pd.testing.assert_frame_equal(
            self.create_dataset._add_one_hot_encoded_versions(df),
            expected_output, check_exact=False, check_like=True,
        )
