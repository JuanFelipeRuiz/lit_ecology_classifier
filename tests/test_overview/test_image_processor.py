import os
from datetime import datetime as dt, timezone
import pytest

import typeguard
from unittest.mock import patch

manager = typeguard.install_import_hook("lit_ecology_classifier.data_overview.utils.image_processing")
import lit_ecology_classifier.data_overview.utils.image_processing as ip


# Test extract_timestamp_from_filename -------------------------------------------------------
def test_extract_timestamp_from_filename():
    """Test correct extraction of a timestamp from a valid filename."""
    file_name = "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg"
    expected_timestamp = dt.strptime("2019-10-08 14:02:52", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )

    assert ip.extract_timestamp_from_filename(file_name) == expected_timestamp

def test_extract_timestamp_from_filename_wrong_timestamp_format():
    """Test extraction of timestamp raises ValueError for invalid filename format."""
    file_name = (
            "SPC-EAWAG-0P5X-1AA7054337290115-3725350526242-001629-055-1224-2176-84-64"
        )

    with pytest.raises(ValueError):
        ip.extract_timestamp_from_filename(file_name)


# Test the different metadata extraction functions ---------------------------------------------

# mock the return value of the timestamp extraction function to make the test independent of the actual timestamp
@patch("lit_ecology_classifier.data_overview.utils.image_processing.extract_timestamp_from_filename", return_value= dt.fromtimestamp(int(15705433729), tz=timezone.utc))
def test_extract_extract_metadata_V1(mock_extract_timestamp):
    """Test extraction of metadata for version 1."""
    image_path = os.path.join("mock", "zooplankton_0p5x", "asplanchna", "training_data", "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg") 
    expected_metadata = {
        "image": "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg",
        "class": "asplanchna",
        "date": dt.fromtimestamp(int(15705433729), tz=timezone.utc),
    }

    assert ip.extract_metadata_V1(image_path) == expected_metadata


# mock the return value of the timestamp extraction function to make the test independent of the actual timestamp
@patch("lit_ecology_classifier.data_overview.utils.image_processing.extract_timestamp_from_filename", return_value= dt.fromtimestamp(int(15705433729), tz=timezone.utc))
def test_extract_metadata_DSPC(mock_extract_timestamp):
    """Test extraction of metadata for DSPC dataset."""
    image_path = os.path.join("mock", "zooplankton_0p5x", "asplanchna",  "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg") 
    expected_metadata = {
        "image": "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg",
        "class": "asplanchna",
        "date": dt.fromtimestamp(int(15705433729), tz=timezone.utc),
    }

    assert ip.extract_metadata_DSPC(image_path) == expected_metadata


# mock the return value of the timestamp extraction function to make the test independent of the actual timestamp
@patch("lit_ecology_classifier.data_overview.utils.image_processing.extract_timestamp_from_filename", return_value= dt.fromtimestamp(int(15705433729), tz=timezone.utc))
def test_extract_metadata_ood(mock_extract_timestamp):
    """Test extraction of metadata for OOD dataset."""
    image_path = os.path.join("mock", "ood", "ood_1", "asplanchna", "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg") 
    expected_metadata = {
        "image": "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg",
        "class": "asplanchna",
        "date": dt.fromtimestamp(int(15705433729), tz=timezone.utc),
        "ood_cell": "ood_1",
    }
    assert ip.extract_metadata_ood(image_path) == expected_metadata

@pytest.fixture
def test_image_processor():
    """Fixture to initialize the ProcessImage class."""
    return ip.ProcessImage()


@pytest.mark.parametrize("dataset,expected_mock,other_mocks", [
        ("zoolake1", "mock_v1", ["mock_dspc", "mock_ood"]),
        ("zoolake2", "mock_dspc", ["mock_v1", "mock_ood"]),
        ("ood", "mock_ood", ["mock_v1", "mock_dspc"])
    ])

@patch('lit_ecology_classifier.data_overview.utils.image_processing.extract_metadata_V1')
@patch('lit_ecology_classifier.data_overview.utils.image_processing.extract_metadata_DSPC')
@patch('lit_ecology_classifier.data_overview.utils.image_processing.extract_metadata_ood')
def test_extractor_called(mock_ood, mock_dspc, mock_v1, dataset, expected_mock, other_mocks):
    """Test if the correct metadata extractor is called based on dataset version."""
    image_path = "test_image.jpeg"
    processor = ip.ProcessImage()
        
    # Create mock mapping
    mock_map = {
        "mock_v1": mock_v1,
        "mock_dspc": mock_dspc,
        "mock_ood": mock_ood
    }
        
    # Execute test
    processor.extract_metadata(image_path, dataset=dataset)
        
    # Assert expected mock was called
    mock_map[expected_mock].assert_called_once_with(image_path)
        
    # Assert other mocks were not called
    for mock_name in other_mocks:
        mock_map[mock_name].assert_not_called()
    
manager.uninstall()