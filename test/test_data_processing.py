import sys
import numpy as np
import unittest
sys.path.append("../source")
from data_processing import DataProcessing


class TestDataProcessing(unittest.TestCase):
    """
    Test class for the DataProcessing class
    """

    def setUp(self):
        """
        Set up the class with test fixtures

        :param target_sr: int, target resample rate
        :param data_processing: class, create an instance of the DataProcessing class
        :param fs: int, audio sample rate
        :param data: np.ndarray, generate random data
        """
        self.target_sr = 8000
        self.data_processing = DataProcessing(self.target_sr)
        self.fs = 44100
        self.data = np.random.rand(self.fs)

    def test_resample_data(self):
        """
        Test the resample_data method of the DataProcessing class
        """
        # resample the data using the DataProcessing instance
        resampled_data = self.data_processing.resample_data(self.fs, self.data)

        # check if the number of samples in the resampled data is equal to the target sample rate
        self.assertEqual(resampled_data.shape[0], self.target_sr)

        # check if the data type of the resampled data is float32
        self.assertEqual(resampled_data.dtype, np.float32)
