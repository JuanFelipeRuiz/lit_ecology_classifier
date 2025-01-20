import pandas as pd
import pytest

from lit_ecology_classifier.checks import duplicates

def test_duplicate_check_duplicate():
    """Test if the function raises a warning when duplicates are present
       and returns the dataframe containing the duplicate info"""

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

    expected_output = pd.DataFrame(
        {
            "sha256": ["hash1", "hash1"],
            "data_set_version": ["1", "1"],
            "count": [2, 2],
            "diffrent_class": [False, False],
            "diffrent_image_name": [True, True],
            "image": ["image1.jpeg", "image2.jpeg"],
            "class": ["A", "A"],
        },
        index=[0, 1],
    )

    with pytest.warns(UserWarning, match="Duplicates found in the dataset: 2"):
        output_df = duplicates.check_duplicates(input_df)

    pd.testing.assert_frame_equal(output_df, expected_output)

def test_duplicate_check_no_duplicate():
    """Test if no warning is raised when no duplicates are present."""
    df = pd.DataFrame(
        {
            "image": ["image1.jpeg", "image2.jpeg"],
            "class": ["A", "B"],
            "data_set_version": ["1", "1"],
            "sha256": ["hash1", "hash2"],
            "date": ["2021-01-01", "2021-01-01"],
        }
    )
    assert duplicates.check_duplicates(df) is None

def test_duplicate_check_different_version():
    """Test if no warning is raised for similar hashes but different dataset versions."""
    df = pd.DataFrame(
        {
            "image": ["image1.jpeg", "image1.jpeg"],
            "class": ["A", "A"],
            "data_set_version": ["1", "2"],
            "sha256": ["hash1", "hash1"],
            "date": ["2021-01-01", "2021-01-01"],
        }
        )
    assert duplicates.check_duplicates(df) is None
