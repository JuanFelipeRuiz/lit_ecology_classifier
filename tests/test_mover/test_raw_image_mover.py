"""
Unit test for the RawImageMover class inside of the lit_ecology_classifier/data/mover/raw_images_mover.py file.
"""



import os
from unittest.mock import patch

import pytest
import pandas as pd


from lit_ecology_classifier.data.mover.raw_images_mover import RawImageMover


class TestRawImageMover:

    # A fixture is a function that is called by pytest before running the individual test functions.
    # It means that the set_up is called before each test function and makes the test functions are
    # independent from each other but the initialization need only to made once.
    @pytest.fixture(autouse=True)
    @patch("lit_ecology_classifier.data.mover.raw_images_mover.OverviewCreator")
    def set_up(self, mock_overview_creator):
        tgt_base_path = os.path.join("test", "tgt", "base", "path")
        self.raw_image_mover = RawImageMover(tgt_base_path, mock_overview_creator)

    # _get_image_paths -----------------------------------------------------------------------

    # create test matrix with version, paths and expected class result
    @pytest.mark.parametrize(
        ("version", "path", "expected_class"),
        [
            (
                "1",
                os.path.join("test", "path", "class1", "train", "image.jpg"),
                "class1",
            ),
            ("2", os.path.join("test", "path", "class2", "image.jpg"), "class2"),
            ("3", os.path.join("test", "path", "class3", "image.jpg"), "class3"),
        ],
    )

    # test _class_finder method with the test matrix.
    # Each row in the test matrix is passed as an argument to the test function
    # and is tested individually.
    def test_class_finder(self, version, path, expected_class):

        assert self.raw_image_mover._class_finder(version, path) == expected_class

    # _image_paths_to_df -----------------------------------------------------------------------

    def test_image_paths_to_df(self):

        # create a dictionary with the image paths as input
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

        # use pd.testing.assert_frame_equal to compare the expected df with the actual df
        pd.testing.assert_frame_equal(
            self.raw_image_mover._image_paths_to_df(image_paths), expected_df
        )

    # test _prepare_tgt_paths -----------------------------------------------------------------------

    # mock the _class_finder method for a independet test
    @patch(
        "lit_ecology_classifier.data.mover.raw_images_mover.RawImageMover._class_finder"
    )
    def test_prepare_tgt_paths(self, mock_classfinder):

        # create a input dataframe with the data_set_version and image_path
        input_df = pd.DataFrame(
            {
                "data_set_version": ["1", "2"],
                "image_path": [
                    os.path.join("path", "path1", "image1"),
                    os.path.join("path", "path2", "image1"),
                ],
            }
        )

        # mock the _class_finder method to return "class1" and "class2" for the two versions
        mock_classfinder.side_effect = ["class1", "class2"]

        expected_df = pd.DataFrame(
            {
                "data_set_version": ["1", "2"],
                "image_path": [
                    os.path.join("path", "path1", "image1"),
                    os.path.join("path", "path2", "image1"),
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
