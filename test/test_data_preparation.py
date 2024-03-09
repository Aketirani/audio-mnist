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
        :param feature_dict_t: dict, time-domain features
        :param feature_dict_f: dict, frequency-domain features
        :param df: pd.DataFrame, sample DataFrame
        """
        self.target_sr = 2
        self.data_preparation = DataPreparation(self.target_sr)
        self.fs = 4
        self.audio_data = np.array([0.1, 0.2, 0.3, 0.4])
        self.fft_data = np.array([1 + 0.0j, -0.2 + 0.2j, -0.2 + 0.0j, -0.2 - 0.2j])
        self.feature_dict_t = {
            "mean_t": 0,
            "std_t": 10,
            "med_t": 20,
            "q25_t": 30,
            "q75_t": 40,
            "min_t": 50,
            "max_t": 60,
            "skew_t": 70,
            "kurt_t": 80,
            "zeroxrate_t": 90,
            "entropy_t": 100,
        }
        self.feature_dict_f = {
            "mean_f": 0,
            "std_f": 10,
            "med_f": 20,
            "q25_f": 30,
            "q75_f": 40,
            "min_f": 50,
            "max_f": 60,
            "skew_f": 70,
            "kurt_f": 80,
            "sfm_f": 90,
            "cent_f": 100,
        }
        self.df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    def test_resample_data(self):
        resampled_data = self.data_preparation.resample_data(self.fs, self.audio_data)
        self.assertEqual(resampled_data.shape[0], self.target_sr)
        self.assertEqual(resampled_data.dtype, np.float32)

    def test_zero_pad(self):
        resampled_data = self.data_preparation.resample_data(self.fs, self.audio_data)
        zero_padded_data = self.data_preparation.zero_pad(resampled_data)
        self.assertEqual(zero_padded_data.shape[0], self.target_sr)

    def test_fft_data(self):
        fft_of_data = self.data_preparation.fft_data(self.audio_data)
        self.assertTrue(np.allclose(fft_of_data, self.fft_data))

    def test_bandpass_filter(self):
        low_threshold = 0.01
        high_threshold = 0.02
        filtered_fft_data = self.data_preparation.bandpass_filter(
            self.fft_data, low_threshold, high_threshold
        )
        frequencies = np.fft.fftfreq(len(filtered_fft_data), 1 / self.target_sr)
        fft_result = np.abs(np.fft.fft(filtered_fft_data))
        for i, f in enumerate(frequencies):
            self.assertLessEqual(fft_result[i], 1e-5)

    def test_feature_creation_time_domain(self):
        features = self.data_preparation.feature_creation_time_domain(self.audio_data)
        self.assertEqual(set(features.keys()), self.feature_dict_t.keys())

    def test_feature_creation_frequency_domain(self):
        features = self.data_preparation.feature_creation_frequency_domain(
            self.fft_data
        )
        self.assertEqual(set(features.keys()), self.feature_dict_f.keys())

    def test_normalize_features(self):
        normalized_features = self.data_preparation.normalize_features(
            self.feature_dict_f
        )
        self.assertEqual(min(normalized_features.values()), 0)
        self.assertEqual(max(normalized_features.values()), 1)
