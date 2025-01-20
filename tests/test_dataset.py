import os
from unittest.mock import patch
import pytest
import pandas as pd
from lit_ecology_classifier.data_overview.overview_creator import OverviewCreator


class TestCreateOverviewDf:
    """Test suite for the CreateOverviewDf class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test case."""
        with patch("os.path.exists", return_value=True):
            self.create_dataset = OverviewCreator()

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
                "version_1": [True, False, True],
                "version_2": [True, True, False],
            }
        )

        pd.testing.assert_frame_equal(
            self.create_dataset._add_one_hot_encoded_versions_and_group_by(df),
            expected_output,
        )