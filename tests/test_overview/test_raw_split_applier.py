import os
from unittest.mock import patch, call
import pandas as pd
import pytest
from lit_ecology_classifier.data_overview.utils.raw_split_applier import _RawSplitApplier


class TestRawSplitApplier:
    """Test suite for the private class `_RawSplitApplier`."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Fixture to initialize the _RawSplitApplier with mock split file paths."""
        self.split_file_paths = {
            "1": {
                "train": "path/to/train.txt",
                "test": "path/to/test.txt",
                "val": "path/to/val.txt"
            },
            "2": "path/to/split.pkl"
        }
        self.raw_split_applier = _RawSplitApplier(self.split_file_paths)

    # _load_split_overview_from_txt -----------------------------------------------------------

    def mock_loadtxt_side_effect(self, filepath, *args, **kwargs):
        """Mock side effect for `numpy.loadtxt`, returning values based on filename."""
        filename = os.path.basename(filepath)
        if "test" in filename:
            return ["p/image_te1.jpeg", "p/image_te2.jpeg", "p/image_te3.jpeg"]
        elif "train" in filename:
            return ["p/image_tr1.jpeg", "p/image_tr2.jpeg", "p/image_tr3.jpeg"]
        elif "val" in filename:
            return ["p/image_val1.jpeg", "p/image_val2.jpeg", "p/image_val3.jpeg"]
        else:
            return []

    @patch("os.path.exists", autospec=True)
    @patch("numpy.loadtxt")
    def test_split_overview_from_txt(self, mock_loadtxt, mock_exists):
        """Test if `_load_split_overview_from_txt` loads splits correctly."""
        # Mock os.path.exists to return True for all files checked in the test
        mock_exists.return_value = True

        # Set the side effect for numpy.loadtxt
        mock_loadtxt.side_effect = self.mock_loadtxt_side_effect

        # Update split file paths for version "1"
        self.raw_split_applier._split_file_paths = {
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

        assert self.raw_split_applier._load_split_overview_from_txt("1") == expected_output

    # _load_split_overview_from_pickle --------------------------------------------------------

    @patch("pandas.read_pickle")
    def test_load_split_overview_from_pickle(self, mock_read_pickle):
        """Test if `_load_split_overview_from_pickle` loads splits correctly from a pickle file."""
        # Update split file paths for version "2"
        self.raw_split_applier._split_file_paths = {"2": {"pickle": "path/to/filenames.pickle"}}

        # Mock the return value of `pandas.read_pickle`
        mock_read_pickle.return_value = [
            ["p/image_tr1.jpeg", "p/image_tr2.jpeg", "p/image_tr3.jpeg"],  # Train split
            ["p/image_te1.jpeg", "p/image_te2.jpeg", "p/image_te3.jpeg"],  # Test split
            ["p/image_val1.jpeg", "p/image_val2.jpeg", "p/image_val3.jpeg"],  # Val split
        ]

        # Expected output based on the mocked return value
        expected_output = {
            "train": ["p/image_tr1.jpeg", "p/image_tr2.jpeg", "p/image_tr3.jpeg"],
            "test": ["p/image_te1.jpeg", "p/image_te2.jpeg", "p/image_te3.jpeg"],
            "val": ["p/image_val1.jpeg", "p/image_val2.jpeg", "p/image_val3.jpeg"],
        }

        # Assert the method correctly loads and maps splits from the pickle
        assert self.raw_split_applier._load_split_overview_from_pickle("2") == expected_output

    # apply_splits ---------------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("version", "should_call_txt", "should_call_pickle"),
        [
            ("1", True, False),  # Version 1 should call txt
            ("2", False, True),  # Version 2 should call pickle
            ("3", False, True),  # Version 3 should call pickle
        ],
    )
    @patch("lit_ecology_classifier.data_overview.utils.raw_split_applier._RawSplitApplier._load_split_overview_from_txt")
    @patch("lit_ecology_classifier.data_overview.utils.raw_split_applier._RawSplitApplier._load_split_overview_from_pickle")
    def test_process_versions_splits_by_version(
        self,
        mock_load_pickle,
        mock_load_txt,
        version,
        should_call_txt,
        should_call_pickle,
    ):
        """Test if the correct function is called for each version of the test matrix."""
        # Mock the `_split_file_paths` attribute
        self.raw_split_applier._split_file_paths = {version: "_"}

        # Call the function
        self.raw_split_applier.apply_splits(df=pd.DataFrame())

        # Assert that the correct function was called
        if should_call_txt:
            mock_load_txt.assert_called_once()
        else:
            mock_load_txt.assert_not_called()

        if should_call_pickle:
            mock_load_pickle.assert_called_once()
        else:
            mock_load_pickle.assert_not_called()

    # _apply_split_columns_to_dataframe --------------------------------------------------------

    @patch("lit_ecology_classifier.data_overview.utils.raw_split_applier._RawSplitApplier._add_split_column")
    def test_apply_split_columns_to_dataframe(self, mock_add_split_column):
        """Test if `_apply_split_columns_to_dataframe` calls `_add_split_column` correctly."""
        # Set up test inputs for the function
        input_image_paths_split = {
            "train": ["p/image_tr1.jpeg", "p/image_tr2.jpeg"],
            "test": ["p/image_te1.jpeg", "p/image_te2.jpeg"],
            "val": ["p/image_val1.jpeg", "p/image_val2.jpeg"],
        }
        input_version = "2"
        df = None

        # Expected calls of the helper function
        expected_calls = [
            call(df=df, image_paths=["p/image_tr1.jpeg", "p/image_tr2.jpeg"], split_name="train_v2"),
            call(df=df, image_paths=["p/image_te1.jpeg", "p/image_te2.jpeg"], split_name="test_v2"),
            call(df=df, image_paths=["p/image_val1.jpeg", "p/image_val2.jpeg"], split_name="val_v2"),
        ]

        # Mock the return value of `_add_split_column`
        mock_add_split_column.return_value = None

        # Call the function
        self.raw_split_applier._apply_split_columns_to_dataframe(
            images_paths_split=input_image_paths_split, version=input_version, df=df
        )

        # Verify the function was called with the correct arguments
        mock_add_split_column.assert_has_calls(expected_calls, any_order=True)
        assert mock_add_split_column.call_count == 3

    # _add_split_column -----------------------------------------------------------------------

    def test_add_split_group_column(self):
        """Test if `_add_split_column` adds the split column correctly."""
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
            self.raw_split_applier._add_split_column(
                df=input_df, image_paths=train_lst, split_name="train_v1"
            ),
            expected_output,
        )
