import unittest

import numpy as np
import pandas as pd

from src.data_preparation import DataPreparation


class TestDataPreparation(unittest.TestCase):
    """
    Test class for the DataPreparation class
    """

    def setUp(self):
        """
        Set up the class with test fixtures

        :param target_sr: int, target sample rate
        :param data_preparation: class, create an instance of the DataPreparation class
        :param fs: int, audio sample rate
        :param audio_data: np.ndarray, audio data
        :param fft_data: np.ndarray, fft data
        :param feature_dict: dict, features
        :param df: pd.DataFrame, sample DataFrame
        """
        self.target_sr = 2
        self.data_preparation = DataPreparation(self.target_sr)
        self.fs = 4
        self.audio_data = np.array([0.1, 0.2, 0.3, 0.4])
        self.fft_data = np.array([1 + 0.0j, -0.2 + 0.2j, -0.2 + 0.0j, -0.2 - 0.2j])
        self.feature_dict = {
            "mean": 0,
            "std": 10,
            "med": 20,
            "q25": 30,
            "q75": 40,
            "min": 50,
            "max": 60,
            "skew": 70,
            "kurt": 80,
            "sfm": 90,
            "cent": 100,
        }
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    def test_resample_data(self):
        """
        Test the resample_data method
        """
        # call resample_data method to get resampled data
        resampled_data = self.data_preparation.resample_data(self.fs, self.audio_data)

        # check if the number of samples in the resampled data is equal to the target sample rate
        self.assertEqual(resampled_data.shape[0], self.target_sr)

        # check if the data type of the resampled data is float32
        self.assertEqual(resampled_data.dtype, np.float32)

    def test_zero_pad(self):
        """
        Test the zero_pad method
        """
        # call resample_data method to get resampled data
        resampled_data = self.data_preparation.resample_data(self.fs, self.audio_data)

        # call zero_pad method to get zero padded data
        zero_padded_data = self.data_preparation.zero_pad(resampled_data)

        # check if the length of zero padded data is equal to the target sample rate
        self.assertEqual(zero_padded_data.shape[0], self.target_sr)

    def test_fft_data(self):
        """
        Test the fft_data method
        """
        # call fft_data method to get FFT of the audio data
        fft_of_data = self.data_preparation.fft_data(self.audio_data)

        # check if the FFT of the audio data is equal to the expected FFT
        self.assertTrue(np.allclose(fft_of_data, self.fft_data))

    def test_bandpass_filter(self):
        """
        Test the bandpass_filter method
        """
        # define the low and high thresholds for the filter
        low_threshold = 0.01
        high_threshold = 0.02

        # call the bandpass_filter method to get the filtered FFT data
        filtered_fft_data = self.data_preparation.bandpass_filter(
            self.fft_data, low_threshold, high_threshold
        )

        # verify that the frequencies outside the threshold range have been filtered out
        frequencies = np.fft.fftfreq(len(filtered_fft_data), 1 / self.target_sr)
        fft_result = np.abs(np.fft.fft(filtered_fft_data))
        for i, f in enumerate(frequencies):
            self.assertLessEqual(fft_result[i], 1e-5)

    def test_feature_creation(self):
        """
        Test the feature_creation method
        """
        # call the feature_creation method to create the features
        features = self.data_preparation.feature_creation(self.fft_data)

        # check if the created feature columns are equal to the expected feature columns
        self.assertEqual(set(features.keys()), self.feature_dict.keys())

    def test_normalize_features(self):
        """
        Test the normalize_features method
        """
        # call the normalized_features method to normalize the features
        normalized_features = self.data_preparation.normalize_features(
            self.feature_dict
        )

        # check if the normalized features are between 0 and 1
        self.assertEqual(min(normalized_features.values()), 0)
        self.assertEqual(max(normalized_features.values()), 1)

    def test_move_column_to_last(self):
        """
        Test the move_column_to_last method
        """
        # column to move to the last position
        column_to_move = 'B'

        # call the move_column_to_last method
        result_df = self.data_preparation.move_column_to_last(self.df.copy(), column_to_move)

        # check if the specified column is moved to the last position
        expected_columns = ['A', 'C', 'B']
        self.assertEqual(result_df.columns.tolist(), expected_columns)

        # check if the values in the column are unchanged
        self.assertTrue((result_df['B'] == self.df['B']).all())
