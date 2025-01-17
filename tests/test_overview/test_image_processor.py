import os
from datetime import datetime as dt, timezone

import pytest
from unittest.mock import patch

from lit_ecology_classifier.data_overview.utils.image_processing import ProcessImage


@pytest.fixture
def image_processor():
    """Fixture to initialize the ProcessImage class."""
    hash_algorithm = "sha256"
    return ProcessImage(hash_algorithm=hash_algorithm)


class TestProcessImages:
    """Test suite for the ProcessImage class."""

    # Test extract_timestamp_from_filename -------------------------------------------------------

    def test_extract_timestamp_from_filename(self, image_processor):
        """Test correct extraction of a timestamp from a valid filename."""
        file_name = "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg"
        expected_timestamp = dt.strptime("2019-10-08 14:02:52", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )

        assert image_processor._extract_timestamp_from_filename(file_name) == expected_timestamp

    def test_extract_timestamp_from_filename_wrong_timestamp_format(self, image_processor):
        """Test extraction of timestamp raises ValueError for invalid filename format."""
        file_name = (
            "SPC-EAWAG-0P5X-1A7054337290115-3725350526242-001629-055-1224-2176-84-64"
        )

        with pytest.raises(ValueError):
            image_processor._extract_timestamp_from_filename(file_name)

    # Test extract_plankton_class_datalake ------------------------------------------------------

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
    def test_class_finder(self, image_processor, version, path, expected_class):
        """Test extraction of the plankton class based on version and path."""
        assert (
            image_processor._extract_plankton_class(version=version, image_path=path)
            == expected_class
        )

    # Test process_image -------------------------------------------------------------------------

    @patch("lit_ecology_classifier.helpers.hashing.HashGenerator.hash_image", return_value="fake_sha256")
    @patch.object(ProcessImage, "_extract_timestamp_from_filename", return_value="fake_timestamp")
    @patch.object(ProcessImage, "_extract_plankton_class", return_value="fake_class")
    def test_process_image_correct_output(
        self, mock_extract_class, mock_hash_image, mock_extract_timestamp, image_processor
    ):
        """Test if `process_image` produces the correct output."""
        image_name = "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg"
        image_path = os.path.join("fake_class", "test", image_name)

        expected_output = {
            "image": image_name,
            "sha256": "fake_sha256",
            "class": "fake_class",
            "data_set_version": "1",
            "date": "fake_timestamp",
        }

        # Call the method under test
        assert (
            image_processor.process_image(image_path=image_path, version="1")
            == expected_output
        )
        mock_hash_image.assert_called_once()
        mock_extract_class.assert_called_once()
        mock_extract_timestamp.assert_called_once()
