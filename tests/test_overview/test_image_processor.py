import os
from datetime import datetime as dt
from datetime import timezone

from unittest.mock import patch


import pytest

from lit_ecology_classifier.data.overview.image_processing import ProcessImage


class TestProcessImages:

    @pytest.fixture(autouse=True)
    def setup(self):
        hash_algorithm = "sha256"
        self.image_processor = ProcessImage(hash_algorithm=hash_algorithm)

    # extract_timestamp_from_filename -------------------------------------------------------------------------------------------

    def test_extract_timestamp_from_filename(self):
        file_name = "SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg"
        expected_timestamp = dt.strptime("2019-10-08 14:02:52", "%Y-%m-%d %H:%M:%S")
        assert self.image_processor._extract_timestamp_from_filename(
            file_name
        ) == expected_timestamp.replace(tzinfo=timezone.utc)

    def test_extract_timestamp_from_filename_wrong_timestamp_format(self):
        file_name = (
            "SPC-EAWAG-0P5X-1A7054337290115-3725350526242-001629-055-1224-2176-84-64"
        )
        with pytest.raises(ValueError):
            self.image_processor._extract_timestamp_from_filename(file_name)

    # extract_plankton_class_datalake -------------------------------------------------------------------------------------------

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
    def test_class_finder(self, version, path, expected_class):
        assert self.image_processor._extract_plankton_class(version=version,image_path = path) == expected_class
    
# process image -------------------------------------------------------------------------------------------

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

        # mock the called functions for a idependet unit test
        with patch.object(
            
            self.image_processor,"_extract_timestamp_from_filename", return_value="fake_timestamp",
        ), patch.object(
            self.image_processor, "hash_image", return_value="fake_sha256"
        ), patch.object(
            self.image_processor, "_extract_plankton_class", return_value="fake_class"
        ):

            assert (
                self.image_processor.process_image(image_path=image_path, version="1")
                == expected_output
            )
