import unittest

import pandas as pd

from src.feature_engineering import FeatureEngineering


class TestFeatureEngineering(unittest.TestCase):
    """
    Test class for the FeatureEngineering class
    """

    def setUp(self):
        """
        Set up the class with test fixtures

        :param feature_engineering: class, an instance of the FeatureEngineering class
        :param df: pd.DataFrame, sample data with gender column
        :param gender_column: list, gender column
        :param threshold: float, correlation threshold above which columns will be removed
        """
        self.feature_engineering = FeatureEngineering()
        self.df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6],
                "col2": [1, 2, 3, 4, 5, 6],
                "col3": [0, 1, 2, 1, 2, 0],
                "col4": [1, 1, 1, 1, 1, 1],
                "gender": ["male", "female", "male", "female", "male", "female"],
            }
        )
        self.gender_column = ["gender"]
        self.threshold = 0.95

    def test_pearson_correlation(self):
        corr_matrix = self.feature_engineering.pearson_correlation(
            self.df, self.gender_column
        )
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(
            corr_matrix.shape,
            (
                self.df.shape[1] - len(self.gender_column),
                self.df.shape[1] - len(self.gender_column),
            ),
        )
        self.assertTrue(
            all(
                col in corr_matrix.columns
                for col in self.df.columns
                if col not in self.gender_column
            )
        )

    def test_remove_constant_columns(self):
        df = self.feature_engineering.remove_constant_columns(
            self.df, self.gender_column
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(
            df.shape[1],
            self.df.shape[1]
            - len(
                [
                    col
                    for col in self.df.columns
                    if self.df[col].nunique() <= 1 and col not in self.gender_column
                ]
            ),
        )
        self.assertTrue(all(col in df.columns for col in self.gender_column))

    def test_remove_correlated_columns(self):
        df_result = self.feature_engineering.remove_correlated_columns(
            self.df,
            self.threshold,
            self.gender_column[0],
        )
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertTrue(self.gender_column[0] in df_result.columns)
        corr_matrix = df_result.drop(self.gender_column[0], axis=1).corr()
        correlated_columns = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    correlated_columns.add(colname)
        self.assertEqual(len(correlated_columns), 0)
        self.assertTrue(df_result.shape[1] < self.df.shape[1])

    def test_categorize_column_values(self):
        result = self.feature_engineering.categorize_column_values(
            self.df, self.gender_column[0]
        )
        self.assertIn(self.gender_column[0], result.columns)
        self.assertEqual(result[self.gender_column[0]].tolist(), [1, 0, 1, 0, 1, 0])
