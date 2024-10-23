import os
from datetime import datetime as dt
from datetime import timezone

import pytest
import pandas as pd
import numpy as np
from unittest.mock import mock_open, patch, call

from lit_ecology_classifier.data.create_overview_data_set import CreateOverviewDf


class TestCreateDataFrame:
    """Testing for the create_dataframe function"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test case"""
        with patch("os.path.exists") as mocked_exists:
            # Mock os.path.exists to return True for required paths
            mocked_exists.return_value = True
            self.create_dataset = CreateOverviewDf()

    # extract_timestamp_from_filename -------------------------------------------------------------------------------------------

    def test_extract_timestamp_from_filename(self):
        file_name = "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg"
        expected_timestamp = dt.strptime("2019-10-08 14:02:52", "%Y-%m-%d %H:%M:%S")
        assert self.create_dataset._extract_timestamp_from_filename(
            file_name
        ) == expected_timestamp.replace(tzinfo=timezone.utc)

    def test_extract_timestamp_from_filename_wrong_timestamp_format(self):
        file_name = (
            "SPC-EAWAG-0P5X-1A7054337290115-3725350526242-001629-055-1224-2176-84-64"
        )
        with pytest.raises(ValueError):
            self.create_dataset._extract_timestamp_from_filename(file_name)

    # extract_plankton_class_datalake -------------------------------------------------------------------------------------------

    def test_extract_plankton_class_datalake_v2(self):
        image_path = os.path.join("class", "image.jpeg")
        expected_class = "class"
        assert (
            self.create_dataset._extract_plankton_class(image_path, "2")
            == expected_class
        )

    def test_extract_plankton_class_datalake_v1(self):
        image_path = os.path.join("class", "training", "image.jpeg")
        expected_class = "class"
        assert (
            self.create_dataset._extract_plankton_class(image_path, "1")
            == expected_class
        )

    # hash_image_sha256 -----------------------------------------------------------------------------------------------------

    @patch("builtins.open", side_effect=PermissionError)
    def test_hash_image_sha256_access_permision_error(self, mock_open):
        with pytest.raises(PermissionError):
            self.create_dataset._hash_image_sha256("invalid_path.jpeg")

    # process_image ---------------------------------------------------------------------------------------------------------

    def test_process_image_correct_output(self):
        image_name = "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg"
        image_path = os.path.join("fake_class", "test", image_name)
        expected_output = {
            "image": image_name,
            "sha256": "fake_sha256",
            "class": "fake_class",
            "data_set_version": "1",
            "date": "fake_timestamp",
        }

        # Mock the extract_timestamp_from_filename method to return 'timestamp'
        with patch.object(
            self.create_dataset,
            "_extract_timestamp_from_filename",
            return_value="fake_timestamp",
        ), patch.object(
            self.create_dataset, "_hash_image_sha256", return_value="fake_sha256"
        ), patch.object(
            self.create_dataset, "_extract_plankton_class", return_value="fake_class"
        ):

            assert (
                self.create_dataset.process_image(image_path=image_path, version="1")
                == expected_output
            )

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

    # load_split_overview_from_txt -----------------------------------------------------------------------------------------------------

    def mock_loadtxt_side_effect(self, filepath, *args, **kwargs):
        """Mock side effect for the loadtxt function.
        Returns different values depending on the filename
        that is passed to the function nploadtxt"""

        filename = os.path.basename(filepath)
        print(f"Mocking loadtxt for file: {filename}")
        if "test" in filename:
            return ["p/image_te1.jpeg", "p/image_te2.jpeg", "p/image_te3.jpeg"]
        elif "train" in filename:
            return ["p/image_tr1.jpeg", "p/image_tr2.jpeg", "p/image_tr3.jpeg"]
        elif "val" in filename:
            return ["p/image_val1.jpeg", "p/image_val2.jpeg", "p/image_val3.jpeg"]
        else:
            return []

    @patch("os.path.exists", autospec=True)
    def test_split_overview_from_txt(self, mock_exists):
        """Test if the function returns the correct output using a mock of the loadtxt function"""

        with patch("numpy.loadtxt") as mock_loadtxt:

            # Mock os.path.exists to return True for all files checked in the test
            mock_exists.return_value = True

            # Set the side effect for numpy.loadtxt to use the custom mock function
            mock_loadtxt.side_effect = self.mock_loadtxt_side_effect

            self.create_dataset._split_file_paths = {
                "1": {
                    "train": "path/to/train_filenames.txt",
                    "test": "path/to/test_filenames.txt",
                    "val": "path/to/val_filenames.txt",
                }
            }

            expected_output = {
                "train": ["p/image_tr1.jpeg", "p/image_tr2.jpeg", "p/image_tr3.jpeg"],
                "test": ["p/image_te1.jpeg", "p/image_te2.jpeg", "p/image_te3.jpeg"],
                "val": ["p/image_val1.jpeg", "p/image_val2.jpeg", "p/image_val3.jpeg"],
            }

            assert (
                self.create_dataset._load_split_overview_from_txt("1")
                == expected_output
            )

    # load_split_overview_from_pickle -----------------------------------------------------------------------------------------------------

    @patch("os.path.exists", autospec=True)
    def test_load_split_overview_from_pickle(self, mock_exists):
        with patch("pandas.read_pickle") as mock_read_pickle:

            self.create_dataset._split_file_paths = {
                "2": {
                    "pickle": "path/to/filenames.pickle",
                }
            }

            mock_read_pickle.return_value = [
                ["p/image_tr1.jpeg", "p/image_tr2.jpeg", "p/image_tr3.jpeg"],
                ["p/image_te1.jpeg", "p/image_te2.jpeg", "p/image_te3.jpeg"],
                ["p/image_val1.jpeg", "p/image_val2.jpeg", "p/image_val3.jpeg"],
            ]

            expected_output = {
                "train": ["p/image_tr1.jpeg", "p/image_tr2.jpeg", "p/image_tr3.jpeg"],
                "test": ["p/image_te1.jpeg", "p/image_te2.jpeg", "p/image_te3.jpeg"],
                "val": ["p/image_val1.jpeg", "p/image_val2.jpeg", "p/image_val3.jpeg"],
            }

            assert (
                self.create_dataset._load_split_overview_from_pickle("2")
                == expected_output
            )

    # add_split_group_column -----------------------------------------------------------------------------------------------------

    def test_add_split_group_column(self):

        input_df = pd.DataFrame(
            {
                "image": [
                    "SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-76.jpeg",
                    "SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-77.jpeg",
                ],
                "class": ["A", "B"],
                "data_set_version": ["1", "1"],
                "sha256": ["hash1", "hash2"],
                "date": ["2021-01-01", "2021-01-01"],
            }
        )

        train_lst = [
            "/home/EAWAG/chenchen/data/Zooplankton/train_data/training_zooplankton_new_220823///cyclops/SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-76.jpeg"
        ]

        expected_output = pd.DataFrame(
            {
                "image": [
                    "SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-76.jpeg",
                    "SPC-EAWAG-0P5X-1656648486404232-61155363907409-004769-280-3640-506-64-77.jpeg",
                ],
                "class": ["A", "B"],
                "data_set_version": ["1", "1"],
                "sha256": ["hash1", "hash2"],
                "date": ["2021-01-01", "2021-01-01"],
                "train_v1": [True, False],
            }
        )

        pd.testing.assert_frame_equal(
            self.create_dataset._add_split_column(
                df=input_df, image_paths=train_lst, split_name="train_v1"
            ),
            expected_output,
        )

    # get_data_splits_by_version -----------------------------------------------------------------------------------------------------

    # create a matrix that contains the version and the expected output for the uni test
    @pytest.mark.parametrize(
        ("version", "should_call_txt", "should_call_pickle"),
        [
            ("1", True, False),  # Version 1 should call txt, no warning
            ("2", False, True),  # Version 2 should call pickle, no warning
            ("3", False, True),  # Version 3 should trigger a warning
        ],
    )

    # create a mock for the load_split_overview_from_txt and load_split_overview_from_pickle functions to check if they are called
    @patch(
        "lit_ecology_classifier.data.create_overview_data_set.CreateOverviewDf._load_split_overview_from_txt"
    )
    @patch(
        "lit_ecology_classifier.data.create_overview_data_set.CreateOverviewDf._load_split_overview_from_pickle"
    )
    def test_process_versions_splits_by_version(
        self,
        mock_function_pickle,
        mock_function_txt,
        version,
        should_call_txt,
        should_call_pickle,
    ):

        with patch.object(self.create_dataset, "_split_file_paths", new={version: "_"}):

            # call the function without expecting a warning. If there is warning, the unit test will also show an warning
            self.create_dataset._process_versions_splits_by_version(df=pd.DataFrame())

            # check if the load_split_overview_from_txt should be called
            if should_call_txt:
                mock_function_txt.assert_called_once()
            else:
                mock_function_txt.assert_not_called()

            if should_call_pickle:
                mock_function_pickle.assert_called_once()
            else:
                mock_function_pickle.assert_not_called()

    # prepare_split_paths -----------------------------------------------------------------------------------------------------
    # create a matrix that contains the version and the expected output for the uni test
    @pytest.mark.parametrize(
        ("version", "should_call_txt", "should_call_pickle", "should_warn"),
        [
            ("1", True, False, False),  # Version 1 should call txt, no warning
            ("2", False, True, False),  # Version 2 should call pickle, no warning
            ("3", False, True, True),  # Version 3 should trigger a warning
            ("Q", False, False, False),  # Version Q should trigger nothing
        ],
    )

    # create a mock for the functions to check if they are called
    @patch(
        "lit_ecology_classifier.data.create_overview_data_set.CreateOverviewDf._prepare_split_paths_from_txt"
    )
    @patch(
        "lit_ecology_classifier.data.create_overview_data_set.CreateOverviewDf._prepare_split_paths_from_pickle"
    )
    def test_prepare_split_paths(
        self,
        mock_function_pickle,
        mock_function_txt,
        version,
        should_call_txt,
        should_call_pickle,
        should_warn,
    ):

        with patch.object(
            self.create_dataset, "zoolake_version_paths", new={version: "_"}
        ):

            # check if it should warn or not
            if should_warn:
                # checks if a warning is raised with the correct message
                with pytest.warns(
                    UserWarning,
                    match="New version, assuming a pickle file for split in the folder",
                ):
                    self.create_dataset._prepare_split_paths()
            else:
                # call the function without expecting a warning. If there is warning, the unit test will also show an warning
                self.create_dataset._prepare_split_paths()

            # check if the load_split_overview_from_txt should be called
            if should_call_txt:
                mock_function_txt.assert_called_once()
            else:
                mock_function_txt.assert_not_called()

            if should_call_pickle:
                mock_function_pickle.assert_called_once()
            else:
                mock_function_pickle.assert_not_called()

    @patch(
        "lit_ecology_classifier.data.create_overview_data_set.CreateOverviewDf._add_split_column"
    )
    def test_function_called_with_correct_columnname(self, mock_function):

        # Set up test inputs
        input_image_paths_split = {
            "train": ["p/image_tr1.jpeg", "p/image_tr2.jpeg"],
            "test": ["p/image_te1.jpeg", "p/image_te2.jpeg"],
            "val": ["p/image_val1.jpeg", "p/image_val2.jpeg"],
        }

        version = "2"
        df = None

        # Expected calls
        expected_calls = [
            call(
                df=df,
                image_paths=["p/image_tr1.jpeg", "p/image_tr2.jpeg"],
                split_name="train_v2",
            ),
            call(
                df=df,
                image_paths=["p/image_te1.jpeg", "p/image_te2.jpeg"],
                split_name="test_v2",
            ),
            call(
                df=df,
                image_paths=["p/image_val1.jpeg", "p/image_val2.jpeg"],
                split_name="val_v2",
            ),
        ]

        # Set return value of the mock function to none since the functions reuses the output as input dataframe
        mock_function.return_value = None

        self.create_dataset._apply_splits_to_dataframe(
            images_paths_split=input_image_paths_split, version=version, df=df
        )

        # Verify the function was called with the correct arguments
        mock_function.assert_has_calls(expected_calls, any_order=True)

        # Ensure it was called exactly three times
        assert mock_function.call_count == 3
