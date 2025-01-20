"""
Test suite for the RawImageMover class inside of the lit_ecology_classifier/data/mover/raw_images_mover.py file.
"""

import os

from unittest.mock import patch
import typeguard 
import pytest
import pandas as pd

manager = typeguard.install_import_hook("lit_ecology_classifier.data_overview.images_copier")
from lit_ecology_classifier.data_overview.images_copier import ImageCopier


class TestRawImageMover:

    # A fixture is a function that is called by pytest before running each individual unit test
    @pytest.fixture(autouse=True)
    @patch("lit_ecology_classifier.data_overview.images_copier.OverviewCreator")
    def set_up(self, mocker):
        """Set up a instance class with and a mock for the overview_class """
        mock_overview_creator = mocker.Mock()
        tgt_base_path = os.path.join("test", "tgt", "base", "path")
        self.raw_image_mover = ImageCopier(tgt_base_path, mock_overview_creator)

    # Use pytest parametrize to do 3 independet test
    @pytest.mark.parametrize(
        ("version", "path", "expected_class"),
        [
            ("1", os.path.join("test", "path", "class1", "train", "image.jpg"), "class1"),
            ("2", os.path.join("test", "path", "class2", "image.jpg"), "class2"),
            ("3", os.path.join("test", "path", "class3", "image.jpg"), "class3"),
        ],
    )
    def test_class_finder(self, version, path, expected_class):
        """Test class finder """
        assert self.raw_image_mover._class_finder(version=version, datapath=path) == expected_class

    def test_image_paths_to_df(self):
        """Test the transformation of images paths to a df"""
        image_paths = {
            "1": ["path/path1/image1", "path/path1/image2"],
            "2": ["path/path2/image1", "path/path2/image2"],
        }

        expected_df = pd.DataFrame(
            {
                "data_set_version": ["1", "1", "2", "2"],
                "image_path": [
                    "path/path1/image1",
                    "path/path1/image2",
                    "path/path2/image1",
                    "path/path2/image2",
                ],
            }
        )

        # use pandas testing tool, since it is a pandas dataframe object to be combared
        pd.testing.assert_frame_equal(
            self.raw_image_mover._image_paths_to_df(image_paths), expected_df
        )

    
    def test_prepare_tgt_paths(self, mocker):
        """ test _prepare_tgt_paths  """

        # Mock the _class_finder method to the test of _prepare_tgt_paths  independet of class finder
        mock_classfinder = mocker.patch(
            "lit_ecology_classifier.data_overview.images_copier.ImageCopier"
        )

        mock_classfinder.side_effect = ["class1", "class2"]

        input_df = pd.DataFrame(
            {
                "data_set_version": ["1", "2"],
                "image_path": [
                    os.path.join("class1", "folder_between", "image1"),
                    os.path.join("path", "class2", "image1"),
                ],
            }
        )

        expected_df = pd.DataFrame(
            {
                "data_set_version": ["1", "2"],
                "image_path": [
                    os.path.join("class1", "folder_between", "image1"),
                    os.path.join("path", "class2", "image1"),
                ],
                "class": ["class1", "class2"],
                "tgt": [
                    os.path.join("test", "tgt", "base", "path", "class1"),
                    os.path.join("test", "tgt", "base", "path", "class2"),
                ],
            }
        )

        pd.testing.assert_frame_equal(
            self.raw_image_mover._prepare_tgt_paths(input_df), expected_df
        )