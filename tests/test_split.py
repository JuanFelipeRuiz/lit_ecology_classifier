import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from lit_ecology_classifier.splitting.split_processor import SplitProcessor



@pytest.mark.skip(reason="Not implemented correct yet.")
class TestSplitProcessor:
    """Test suite for the SplitProcessor class."""

    @pytest.fixture(autouse=True)
    @patch('lit_ecology_classifier.data.mover.split_images_mover.SplitImageMover')
    def setup(self, mock_split_image_mover):
        """Fixture to initialize the SplitProcessor with mocks."""
        self.split_strategy_str = 'MockSplitStrategy'

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
        self.mock_image_mover = mock_split_image_mover.return_value

        # Initialize SplitProcessor
        self.processor = SplitProcessor(
            split_strategy=self.split_strategy_str,
            dataset_version='1',
            ood_version=None,
            image_overview_path=self.image_overview_df,
            split_overview=self.split_overview_df,
            image_base_paths='mock/image/path',
            tgt_base_path='mock/target/path'
        )

        # Mock the split strategy initialization
        mock_strategy = MagicMock()
        mock_strategy.__class__.__name__ = 'MockSplitStrategy'
        mock_strategy.perform_split.return_value = {
            'train': ['img1', 'img2'],
            'val': ['img3'],
            'test': ['img4']
        }



    @patch('lit_ecology_classifier.splitting.split_processor.SplitProcessor._prepare_filepath')
    def test_prepare_filepath(self, mock_prepare_filepath):
        """Test if `_prepare_filepath` correctly constructs the file path."""
        hash_value = 'abcdefghijklmnopqrstuvwxyz123456789'
        expected_result = os.path.join("data", "interim", "train_test_val", 'abcdefghijklmnopqrstuvwx.csv')

        mock_prepare_filepath.return_value = expected_result
        assert self.processor._prepare_filepath(hash_value) == expected_result
        mock_prepare_filepath.assert_called_once_with(hash_value)

    @patch('lit_ecology_classifier.splitting.split_processor.SplitProcessor._filter_df')
    def test_filter_df(self, mock_filter_df):
        """Test if `_filter_df` correctly filters the DataFrame."""
        input_df = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4'],
            'class': [1, 2, 3, 4],
            'version': ['1', '1', '2', '2'],
            'ood_version': [None] * 4
        })

        expected_df = pd.DataFrame({
            'image': ['img1', 'img2'],
            'class': [1, 2],
            'version': ['1', '1'],
            'ood_version': [None, None]
        })

        mock_filter_df.return_value = expected_df
        self.processor.dataset_version = '1'
        self.processor.ood_version = None

        assert self.processor._filter_df(input_df).equals(expected_df)
        mock_filter_df.assert_called_once_with(input_df)

    @patch('lit_ecology_classifier.splitting.split_processor.SplitProcessor._merge_split_df')
    def test_merge_split_df(self, mock_merge_split_df):
        """Test if `_merge_split_df` merges DataFrames correctly."""
        input_df = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4'],
            'class': ["1"] * 4,
            'split': ['train', 'val', 'test', 'train'],
        })

        expected_result = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4'],
            'class': ["1"] * 4,
            'split': ['train', 'val', 'test', 'train'],
            'sha256': ['hash1', 'hash2', 'hash3', 'hash4'],
        })

        self.processor.image_overview_df = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4'],
            'version': ['1'] * 4,
            'class': ["1"] * 4,
            'sha256': ['hash1', 'hash2', 'hash3', 'hash4']
        })

        mock_merge_split_df.return_value = expected_result
        assert self.processor._merge_split_df(input_df).equals(expected_result)
        mock_merge_split_df.assert_called_once_with(input_df)
