"""
Process the image to extract metadata and hash value.

The metadata extracted from the image are: 
    - Image path: The path to the image file
    - Image: The name of the image file
    - Hash: The hash value of the image
    - Class: The plankton class extracted from the image path
    - Dataset version: The version of the dataset
    - Date: The date extracted from the image filename
"""

import logging
import os
from typing import Union, Optional
from datetime import datetime as dt, timezone
import warnings

from lit_ecology_classifier.helpers.hashing import HashGenerator

logging.basicConfig(level=logging.INFO)


class ProcessImage:
    """Process the image to extract metadata and calculate the hash value of the image.

    Attributes:
        metadata_extractor_mapping (dict): Mapping of dataset versions to metadata extractor functions.
    """

    def __init__(self):
        """Initialize the ImageProcessor with the given hash algorithm."""

        self.metadata_extractor_mapping = {
            "zoolake1": extract_metadata_V1,
            "zoolake2": extract_metadata_DSPC,
            "ood": extract_metadata_ood,
            "default": extract_metadata_DSPC,
        }

    def extract_metadata(self, image_path: str, dataset: str) -> dict:
        """Calls the metadata extractor function based on the dataset version.

        Args:
            image_path : Path to the image file
            dataset: Version of the dataset

        Returns:
            A dictionary containing the image metadata. 

        Raises:
            Exception: If the image cannot be processed
        """

        # lower case the dataset version to ensure case insensitivity
        dataset = dataset.lower()
        
        # get the metadata extractor function based on the version
        extractor = self.metadata_extractor_mapping.get(dataset, self.metadata_extractor_mapping["default"])

        try:
            metadata_dict = extractor(image_path)
        except Exception as e:
            logging.error("Error extracting metadata from %s: %s",image_path, e)
            raise ValueError(f"Error extracting metadata from {image_path}: {e}")

        return metadata_dict


    def process_image(self, dataset: str, image_path) -> dict:
        """Process a single image. Extract the metadata and calculate the hash value of the image.

        Args:
            image_path: Path to the image file
            dataset: Version of the dataset

        Returns:
            dict: A dictionary containing the image metadata and hashes.

        Raises:
            Exception: If the image cannot be processed.
        """

        # extract the metadata from the image
        image_metadata = self.extract_metadata(image_path, dataset)

        # add the image hash to the metadata dictionary
        image_metadata["hash"] = HashGenerator.hash_image(image_path)

        image_metadata["image_path"] = image_path

        # add the dataset version to the metadata dictionary
        image_metadata["dataset"] = dataset

        return image_metadata


def extract_timestamp_from_filename(image_path: str) -> dt:
    """Extract the timestamp from the image filename and convert it to a datetime object.

    The timestamp (without miliseconds) is expected to be at a fixed position
    in the filename (characters 15-25). This function will extract those characters,
    convert them to an integer timestamp, and return a UTC aware datetime object.

    Args:
        image_path : Path to the image file as string to extract the timestamp from

    Returns:
           A timestamp extracted from the filename as a datetime object with UTC as timezone

    Raises:
        ValueError: If the extracted value cannot be converted to a timestamp
    """

    try:
        # Extract the image name from the path
        image_name = os.path.basename(image_path)

        # Extract the timestamp part and keep only the first 10 characters
        # (ignoring mili seconds)
        timestamp_str = image_name.split("-")[3][:10]

        # return the timestamp as a datetime object with UTC as timezone
        return dt.fromtimestamp(int(timestamp_str), tz=timezone.utc)

    except IndexError as ie:
        raise ValueError(
            f"Error extracting timestamp: Failed slicing timestamp from '{image_name}:{ie}'"
        ) from ie

    except Exception as e:
        raise ValueError(
            f"Error extracting and creating timestamp from {image_path}: {e}"
        ) from e


def extract_metadata_V1(image_path: str) -> dict:
    """Extract needed metadata from the image filename for version 1 of the dataset.

    Differnece betweend the dataset version 2 is that the class is in the second parent directory
    instead of the first parent directory as usually seen in computer vision datasets.

    Args:
        image_path : Path to the image file as string to extract the metadata from

    Returns:
        A dictionary containing the metadata extracted from the image filename.
    """

    return {
        "image": os.path.basename(image_path),
        # The plankton class is the second parent directory of the image file in ZooLake1
        "class": os.path.basename(os.path.dirname(os.path.dirname(image_path))),
        "date": extract_timestamp_from_filename(image_path)
    }

def extract_metadata_DSPC(image_path: str) -> dict:
    """Ectract needed metadata from the image filename for SPC dataset.

    It is the default extractor for the SPCS Aquascope dataset.
    Args:
        image_path : Path to the image file as string to extract the metadata from
    Returns:
        A dictionary containing the metadata extracted from the image filename.
    """

    return {
        "image": os.path.basename(image_path),
        # In every further dataset version, the class is the first parent directory of the image file
        "class": os.path.basename(os.path.dirname(image_path)),
        "date": extract_timestamp_from_filename(image_path)
    }

def extract_metadata_ood(image_path: str) -> dict:
    """Extract needed metadata from the image filename for OOD dataset.

    It is the default extractor for the OOD dataset.
    Args:
        image_path : Path to the image file as string to extract the metadata from
    Returns:
        A dictionary containing the metadata extracted from the image filename.
    """

    return {
        "image": os.path.basename(image_path),
        "class": os.path.basename(os.path.dirname(image_path)),
        "date": extract_timestamp_from_filename(image_path),
        "ood_cell": os.path.basename(os.path.dirname(os.path.dirname(image_path))),
    }


if __name__ == "__main__":
    import pprint

    # Provide the path to the image and the version of the dataset
    image_path = "data/ZooLake1/zooplankton_0p5x/aphanizomenon/training_data/SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg"
    version = "1"

    processor = ProcessImage()
    image_metadata = processor.process_image(version, image_path)

    # pretty print the image metadata for the display
    pprint.pp(image_metadata)
