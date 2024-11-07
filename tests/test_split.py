import hashlib
import os
import unittest
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from lit_ecology_classifier.splitting.split import SplitProcessor


# 'lit_ecology_classifier.data.mover.split_images_mover.SplitImageMover'
class TestSplitProcessor(unittest.TestCase):
    def setUp(self):
        # Pass split_strategy as a string
        self.split_strategy_str = 'MockSplitStrategy'

        # Sample DataFrames
        self.image_overview_df = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4'],
            'class': [1, 2, 3, 4],
            'hash': ['hash1', 'hash2', 'hash3', 'hash4'],
            'version': ['v1.0'] * 4,
            'ood_version': [None] * 4
        })

        self.split_overview_df = pd.DataFrame(columns=[
            'dataset_version',
            'split_method',
            'ood_version',
            'combined_split_hash',
            'description'
        ])

        # Mock SplitImageMover
        with patch('lit_ecology_classifier.data.mover.split_images_mover.SplitImageMover') as MockImageMover:
            self.mock_image_mover = MockImageMover.return_value

            # Initialize SplitProcessor
            self.processor = SplitProcessor(
                split_strategy=self.split_strategy_str,
                dataset_version='v1.0',
                ood_version=None,
                image_overview_path=self.image_overview_df,
                split_overview=self.split_overview_df,
                image_base_paths='mock/image/path',
                tgt_base_path='mock/target/path'
            )

        # Since we're passing split_strategy as a string, we need to mock the strategy initialization
        with patch.object(self.processor, '_init_split_strategy') as mock_init_strategy:
            mock_init_strategy.return_value = (
                self.split_strategy_str,
                self.create_mock_split_strategy()
            )
            # Re-initialize split_strategy attributes
            self.processor.split_strategy_str, self.processor.split_strategy = \
                self.processor._init_split_strategy(self.split_strategy_str)
    
    def create_mock_split_strategy(self):
        mock_strategy = MagicMock()
        mock_strategy.__class__.__name__ = 'MockSplitStrategy'
        mock_strategy.perform_split.return_value = {
            'train': ['img1', 'img2'],
            'val': ['img3'],
            'test': ['img4']
        }
        return mock_strategy


    def test_prepare_filepath(self):
        hash_value = 'abcdefghijklmnopqrstuvwxyz123456789'
        excepted_result = os.path.join("data", "interim", "train_test_val", 'abcdefghijklmnopqrstuvwx.csv')
        assert self.processor._prepare_filepath(hash_value) == excepted_result


    def test_filter_df(self):

        input_df = pd.DataFrame({ 
            'image': ['img1', 'img2', 'img3', 'img4'],
            'class': [1, 2, 3, 4],
            'version': ['1','1', '2', '2'],
            'ood_version': [None] * 4
        })

        expected_df = pd.DataFrame({
            'image': ['img1', 'img2'],
            'class': [1, 2],
            'version': ['1','1'],
            'ood_version': [None, None]
        })

        # mock the attribute value of dataset_version to 1
        # mock the attribute value of ood_version to None
        self.processor.dataset_version = '1'
        self.processor.ood_version = None

        pd.testing.assert_frame_equal(self.processor._filter_df(input_df), expected_df)


    def test_merge_split_df(self):
        input_df = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4'],
            'class': ["1"] * 4,
            'split': ['train', 'val', 'test', 'train'],
        })


        self.processor.image_overview_df = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4'],
            'version': ['1'] * 4,
            'class': ["1"] * 4,
            'sha256' : ['hash1', 'hash2', 'hash3', 'hash4']
        })

        excepted_result = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4'],
            'class': ["1"] * 4,
            'split': ['train', 'val', 'test', 'train'],
            'sha256': ['hash1', 'hash2', 'hash3', 'hash4'],
        })

        pd.testing.assert_frame_equal(self.processor._merge_split_df(input_df), excepted_result)


 
    

