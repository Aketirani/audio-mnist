import unittest

import pandas as pd

from src.data_splitting import DataSplitting


class TestDataSplitting(unittest.TestCase):
    """
    Test class for the DataSplitting class
    """

    def setUp(self):
        """
        Set up the class with test fixtures

        :param test_size: float, proportion of data to be used for the test set
        :param val_size: float, proportion of data to be used for the validation set
        :param data_split: class, create an instance of the DataSplitting class
        :param train_df: pd.DataFrame, dataframe for training data
        :param val_df: pd.DataFrame, dataframe for validation data
        :param test_df: pd.DataFrame, dataframe for test data
        :param target_column: str, the name of the target column
        """
        self.test_size = 0.1
        self.val_size = 0.1
        self.data_split = DataSplitting()
        self.train_df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6],
                "col2": [1, 2, 3, 4, 5, 6],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        self.val_df = pd.DataFrame(
            {
                "col1": [7, 8, 9, 10, 11, 12],
                "col2": [7, 8, 9, 10, 11, 12],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        self.test_df = pd.DataFrame(
            {
                "col1": [13, 14, 15, 16, 17, 18],
                "col2": [13, 14, 15, 16, 17, 18],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        self.target_column = "label"

    def test_split(self):
        combined_df = pd.concat([self.train_df, self.val_df, self.test_df])
        train_df, val_df, test_df = self.data_split.split(
            combined_df, "label", self.test_size, self.val_size
        )
        self.assertEqual(
            train_df["label"].value_counts(normalize=True).values.tolist(),
            combined_df["label"].value_counts(normalize=True).values.tolist(),
        )
        self.assertEqual(
            val_df["label"].value_counts(normalize=True).values.tolist(),
            combined_df["label"].value_counts(normalize=True).values.tolist(),
        )
        self.assertEqual(
            test_df["label"].value_counts(normalize=True).values.tolist(),
            combined_df["label"].value_counts(normalize=True).values.tolist(),
        )

    def test_prepare_data(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_split.prepare_data(
            self.train_df, self.val_df, self.test_df, self.target_column
        )
        self.assertEqual(
            X_train.shape, (len(self.train_df), len(self.train_df.columns) - 1)
        )
        self.assertEqual(y_train.shape, (len(self.train_df),))
        self.assertEqual(X_val.shape, (len(self.val_df), len(self.val_df.columns) - 1))
        self.assertEqual(y_val.shape, (len(self.val_df),))
        self.assertEqual(
            X_test.shape, (len(self.test_df), len(self.test_df.columns) - 1)
        )
        self.assertEqual(y_test.shape, (len(self.test_df),))
        pd.testing.assert_frame_equal(
            X_train, self.train_df.drop(columns=[self.target_column])
        )
        pd.testing.assert_series_equal(y_train, self.train_df[self.target_column])
        pd.testing.assert_frame_equal(
            X_val, self.val_df.drop(columns=[self.target_column])
        )
        pd.testing.assert_series_equal(y_val, self.val_df[self.target_column])
        pd.testing.assert_frame_equal(
            X_test, self.test_df.drop(columns=[self.target_column])
        )
        pd.testing.assert_series_equal(y_test, self.test_df[self.target_column])
