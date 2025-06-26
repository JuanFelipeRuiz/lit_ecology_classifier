import pandas as pd
import numpy as np
import pytest
import typeguard


manager = typeguard.install_import_hook("lit_ecology_classifier.splitting.split_search")
from lit_ecology_classifier.splitting.base_split_processor import BaseSplitProcessor
from lit_ecology_classifier.splitting import split_search

#"""Test suite for helper functions in split_search module."""
    
@pytest.mark.parametrize("input_value,expected_output", [
        ("test", {"test"}),  # Simple string
        ("a,b,c", {"a", "b", "c"}),  # Comma-separated string
        ("  a, b  , c  ", {"a", "b", "c"}),  # String with whitespace
        ([1, 2, 3], {"1", "2", "3"}),  # List of numbers
        (True, {"True"}),  # Boolean
        (1.5, {"1.5"}),  # Float
        ([], set()),  # Empty list
        ("", set()),  # Empty string
        (np.nan, set()),  # NaN value
    ])

def test_normalize_value(input_value, expected_output):
    """Test normalize_value function with different input types."""
    result = split_search.normalize_value(input_value)
    assert result == expected_output

class TestFindMatchingArgs:
    """Test suite for find_matching_args function.
    
    Summarised in one class to keep the tests organized and to use fixtures effectively.
    """

    @pytest.fixture
    def base_row(self):
        """Create a basic row for testing."""
        return pd.Series({
            "split_test_size": 0.2,
            "filter_version": "v1",
            "split_strategy": "RandomSplit"
        })

    @pytest.fixture
    def base_args(self):
        """Create basic expected args for testing."""
        return {
            "split_test_size": 0.2,
            "filter_version": "v1",
            "split_strategy": "RandomSplit"
        }

    def test_exact_match(self, base_row, base_args):
        """Test if exact matching arguments return True."""
        columns = base_row.index
        assert split_search.find_matching_args(base_row, base_args,  columns_to_ignore=[]) 

    def test_multiple_vs_single_value(self, base_row, base_args):
        """Test if multiple values in a single column return False."""
        multiple_args = base_args.copy()
        multiple_args["split_test_size"] = "0.2,0.3"
        columns = base_row.index
        assert split_search.find_matching_args(base_row, multiple_args,  columns_to_ignore=[]) == False


    def test_nan_value(self, base_row, base_args):
        """Test if empty values in the row are treated as empty sets."""
        nan_args = base_args.copy()
        nan_args["split_test_size"] = []

        base_row["split_test_size"] = ""

        columns = base_row.index
        assert split_search.find_matching_args(base_row, nan_args,  columns_to_ignore=[]) == True

    def test_mismatch(self, base_row, base_args):
        """Test if mismatched arguments return False."""
        different_args = base_args.copy()
        different_args["split_test_size"] = 0.3
        columns = base_row.index
        assert split_search.find_matching_args(base_row, different_args,  columns_to_ignore=[]) == False

    def test_partial_match(self, base_row):
        """Test if partial arguments (subset) return False."""
        partial_args = {"split_test_size": 0.2}

        assert split_search.find_matching_args(base_row, partial_args, columns_to_ignore=[]) == False

    def test_extra_columns(self, base_row, base_args):
        """Test if extra columns in args return False."""
        extra_args = base_args.copy()
        extra_args["new_value"] = "xyz"
        columns = base_row.index
        assert split_search.find_matching_args(base_row, extra_args, columns_to_ignore=[]) == False

    @pytest.mark.parametrize("empty_value1,empty_value2", [
    ([], ""),           # empty list vs empty string
    ("", []),           # empty string vs empty list
    (np.nan, ""),       # nan vs empty string
    ("", np.nan),       # empty string vs nan
    ([], np.nan),       # empty list vs nan
    (np.nan, []),        # nan vs empty list
    ])

    def test_empty_values_match(self, empty_value1, empty_value2):
        """Test if different types of empty values are treated as equivalent."""
        # Create a Series directly instead of DataFrame conversion
        split_entry = pd.Series({
            "split_test_size": empty_value1
        })

        entry_args = {
            "split_test_size": empty_value2
        }

        columns = split_entry.index
        result = split_search.find_matching_args(
            split_entry,
            entry_args,
            
            columns_to_ignore=[]
        )
        
        assert result == True, f"Failed matching {empty_value1} with {empty_value2}"

manager.uninstall