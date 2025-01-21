"""Test suite for the argparsing""" 

import json
import tempfile

from unittest.mock import patch
import pytest
import typeguard



manager = typeguard.install_import_hook(packages="lit_ecology_classifier.helpers.argparser")
import lit_ecology_classifier.helpers.argparser  as lit_parser




def test_load_dict_with_empty_string():
    """Test when input is an empty string."""
    result = lit_parser.load_dict("")
    assert result == {}

def test_load_dict_with_none():
    """Test correct return when input is None."""
    result = lit_parser.load_dict(None)
    assert result == {}

def test_load_dict_with_missing_json_file():
    """Test when input is a path to a missing JSON file."""
    missing_file_path = "nonexistent.json"
    with pytest.raises(lit_parser.argparse.ArgumentTypeError, match=f"{missing_file_path} file not found."):
        lit_parser.load_dict(missing_file_path)


def test_load_dict_with_invalid_file_extension():
    """Test when input is a path to a file with an invalid extension."""
    invalid_file_path = "file.txt"
    with pytest.raises(lit_parser.argparse.ArgumentTypeError, match=f"{invalid_file_path} is not a path to a JSON file or dict containing the args."):
        lit_parser.load_dict(invalid_file_path)

