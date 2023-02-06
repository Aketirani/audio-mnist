import sys
import pandas as pd
import unittest
sys.path.append("../src")
from data_split import DataSplit


class TestDataSplit(unittest.TestCase):
    """
    Test class for the DataSplit class
    """
    def setUp(self):
        """
        Set up the class with test fixtures

        :param test_size: float, proportion of data to be used for the test set
        :param val_size: float, proportion of data to be used for the validation set
        :param data_split: class, create an instance of the DataSplit class
        :param df: pd.DataFrame, dataset
        """
        self.test_size = 0.1
        self.val_size = 0.1
        self.data_split = DataSplit(self.test_size, self.val_size)
        self.df = pd.DataFrame({'col1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                                'col2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                                'col3': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    def test_split(self):
        """
        Test the split method
        """
        # Split the dataframe
        train_df, val_df, test_df = self.data_split.split(self.df, 'col3')

        # check if the size of each split dataframe are 80/10/10
        self.assertEqual(train_df.shape[0], int(len(self.df) * (1-self.val_size-self.test_size)))
        self.assertEqual(val_df.shape[0], int(len(self.df) * self.val_size))
        self.assertEqual(test_df.shape[0], int(len(self.df) * self.test_size))

        # check if the ratio of target variable match in each split
        self.assertEqual(train_df['col3'].value_counts(normalize=True).values.tolist(), self.df['col3'].value_counts(normalize=True).values.tolist())
        self.assertEqual(val_df['col3'].value_counts(normalize=True).values.tolist(), self.df['col3'].value_counts(normalize=True).values.tolist())
        self.assertEqual(test_df['col3'].value_counts(normalize=True).values.tolist(), self.df['col3'].value_counts(normalize=True).values.tolist())
