"""
Process the image to extract metadata and hash value.
"""

import logging
import os
from datetime import datetime as dt, timezone

from ...helpers.hashing import HashGenerator

logging.basicConfig(level=logging.INFO)


class ProcessImage:
    """Process the image to extract metadata and calculate the hash value.

    It is used to process the image to extract metadata like timestamp and plankton class
    from the filename. Additionally, it calculates the hash value of the image using
    the given hash algorithm to provide a unique identifierfor each image contained in the dataset.

    Attributes:
        hash_algorithm (str): Hash algorithm to use for hashing images. Defaults to "sha256".

    Returns:
        dict: Dictionary containing the image metadata and calculated hash. Example:
                {
                    "image": "SPC-EAWAG-0P5X-1570543372901157-3725350526242...",
                    "sha256": "a957e3fb302aa924ea62f25b436893151640dc05f761....",
                    "class": "aphanizomenon",
                    "data_set_version": "1",
                    "date": "2019-10-08 14:02:52+00:00"
                }

    """

    def __init__(self, hash_algorithm: str = "sha256"):
        """Initialize the ImageProcessor with the given hash algorithm.

        Args:
            hash_algorithm (str, optional): Hash algorithm to use for hashing images.
                                            Defaults to "sha256".
        """
        self.hash_algorithm = hash_algorithm
    
    def _extract_timestamp_from_filename(self, image_path: str) -> dt:
        """Extract the timestamp from the image filename and convert it to a datetime object.

        The timestamp (without miliseconds) is expected to be at a fixed position
        in the filename (characters 15-25). This function will extract those characters,
        convert them to an integer timestamp, and return a UTC aware datetime object.

        Args:
            image_path (str): Path to the image file as string to extract the timestamp from

        Returns:
           dt: Timestamp extracted from the filename as a datetime object with UTC as timezone

        Raises:
            ValueError: If the extracted value cannot be converted to a timestamp
        """

        try:

            # Extract the image name from the path
            image_name = os.path.basename(image_path)

            # Extract the timestamp part and keep only the first 10 characters
            # (ignoring mili seconds)
            timestamp_str = image_name[15:25]

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

    def _extract_plankton_class(self, image_path: str, version: str) -> str:
        """Extract  plankton class from the image path based on version.

        The extraction method is based on the different dataset versions:
        - For version 1, the plankton class is extracted from the grandparent directory,
            since there is an additional 'training' folder .
        - For other versions, the plankton class is extracted from the immediate parent directory.

        Args:
            image_path (str): Path to the image file
            version (str): Version of the ZooLake dataset as string

        Returns:
            str: The plankton class name
        """
        try:
            if version == "1":

                # Get the second parent / grandparent directory for ZooLake version 1
                return os.path.basename(os.path.dirname(os.path.dirname(image_path)))

            # get the parent directory for each other ZooLake version
            return os.path.basename(os.path.basename(os.path.dirname(image_path)))

        except Exception as e:
            logging.error("Error extracting plankton class from %s: %s", image_path, e)
            raise ValueError(
                f"Error extracting plankton class from {image_path}: {e}"
            ) from e

    def process_image(self, version, image_path) -> dict:
        """Process a single image. Extract the metadata and calculate the hash value of the image.

        Args:
            version_path_tuple (tuple) : tuple containing
                image_path (str): Path to the image file
                version (str): Version of the ZooLake dataset as string for metadata

        Returns:
            dict: Dictionary containing the image metadata and hashes. Example:
                {
                    "image": "SPC-EAWAG-0P5X-1570543372901157-3725350526242...",
                    "sha256": "a957e3fb302aa924ea62f25b436893151640dc05f761...",
                    "class": "aphanizomenon",
                    "data_set_version": "1",
                    "date": "2019-10-08 14:02:52+00:00"
                }

        Raises:
            Exception: If the image cannot be processed
        """

        image_date = self._extract_timestamp_from_filename(image_path)
        plankton_class = self._extract_plankton_class(image_path, version)
        image_hash = HashGenerator.hash_image(image_path, self.hash_algorithm)

        image_metadata = {
            "image": os.path.basename(image_path),
            str(self.hash_algorithm): image_hash,
            "class": plankton_class,
            "data_set_version": version,
            "date": image_date,
        }

        return image_metadata
