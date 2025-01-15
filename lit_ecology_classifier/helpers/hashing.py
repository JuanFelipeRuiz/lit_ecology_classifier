import hashlib
import logging

import pandas as pd
from PIL import Image
import imagehash

class HashGenerator:
    @staticmethod
    def sha256_from_list(hash_list):
        """Hash a list of hashes using SHA256.

        Sorts the hash list based on the hash values and concatenates the hashes to a single
        string The concatenated string is hashed using SHA256.

        Args:
            hash_list (list): List of hashes to concatenate and hash.

        Returns:
            str: SHA256 hash of the concatenated hash list.
        """
        sorted_hashes = sorted(hash_list)
        concatenated = "".join(sorted_hashes)
        return hashlib.sha256(concatenated.encode()).hexdigest()

    @staticmethod
    def generate_hash_dict_from_split(df : pd.DataFrame,
                                      col_to_hash = "hash256", 
                                      group_by_col = "split"

                                      ) -> list[str]:
        """Generate a hash for each value inside the split column.

        Args:
            df (Dataframe): Contains the columns hash and split.
            col_to_hash (str): Column with values to hash.

        Returns:
            dictionary containing the image hashes with the corresponding split as key. Example:
            {
                "train": hash_of_train_data,
                "val": hash_of_val_data,
                "test": hash_of_test_data
            }
        """

        hash_dict = (
            df.groupby(group_by_col)[col_to_hash].apply(HashGenerator.sha256_from_list).to_dict()
        )

        return hash_dict
    
    @staticmethod
    def hash_image(image_path: str, hash_algorithm: str = "256") -> str:
        """Calculate the hash from the binary data of the image using the given hash algorithm.

        Args:
            image_path (str): Path to the image file as string

        Returns:
            str: Hash value of the image as string

        Raises:
            PermissionError: If the image cannot be read due to permission issues
            Exception: If the image cannot be hashed due to other issues
        """
        try:

            with Image.open(image_path) as img:

                #  if image_path ends with .jpeg or .png
                if img.format not in ["JPEG", "PNG"]:
                    logging.error(
                        "Error hashing %s with SHA256: Invalid file format %s",
                        image_path,
                        img.format,
                    )
                    raise ValueError(
                        f"Error hashing {image_path} with SHA256: Invalid file format {img.format}"
                    )

                if hash_algorithm == "sha256":
                    img_data = img.tobytes()
                    return hashlib.sha256(img_data).hexdigest()
                if hash_algorithm == "phash":
                    return str(imagehash.phash(img))

                logging.error("Unsupported hash algorithm: %s", hash_algorithm)
                raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

        except PermissionError as pe:
            logging.error("Permission denied when accessing %s: %s", image_path, pe)
            raise PermissionError(
                f"Permission denied when accessing {image_path}:{pe}"
            ) from pe

        except Exception as e:
            logging.error("Error hashing image %s with SHA256: %s", image_path, e)
            raise Exception(f"Error hashing {image_path} with SHA256: {e}") from e
