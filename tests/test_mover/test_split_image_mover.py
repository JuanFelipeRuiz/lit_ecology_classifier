
import os
from unittest.mock import patch

import pytest
import pandas as pd


from lit_ecology_classifier.data.mover.split_images_mover import SplitImageMover


class SplitImageMover:

    # A fixture is a function that is called by pytest before running the individual test functions.
    # It means that the set_up is called before each test function and makes the test functions are
    # independent from each other but the initialization need only to made once.
    @pytest.fixture(autouse=True)
    def set_up(self):
        image_base_paths = os.path.join("data", "interim", "ZooLake")
        tgt_path = os.path.join("data", "processed")
        self.raw_image_mover = SplitImageMover
