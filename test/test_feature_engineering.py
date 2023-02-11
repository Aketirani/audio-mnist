import sys
import pandas as pd
import unittest
sys.path.append("../src")
from feature_engineering import FeatureEngineering


class TestFeatureEngineering(unittest.TestCase):
    """
    Test class for the FeatureEngineering class
    """
    def setUp(self):
        """
        Set up the class with test fixtures

        :param feature_engineering: class, an instance of the FeatureEngineering class
        :param df: pd.DataFrame, sample data with gender column
        :param columns_to_leave_out: list, list of column names to be excluded from correlation calculation but kept in the final output
        :param threshold: float, correlation threshold above which columns will be removed
        """
        self.feature_engineering = FeatureEngineering()
        self.df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6],
                                        'col2': [1, 2, 3, 4, 5, 6],
                                        'col3': [0, 1, 2, 1, 2, 0],
                                        'col4': [1, 1, 1, 1, 1, 1],
                                        'gender': ['male', 'female', 'male', 'female', 'male', 'female']})
        self.columns_to_leave_out = ['gender']
        self.threshold = 0.95

    def test_pearson_correlation(self):
        """
        Test the pearson_correlation method
        """
        # call the pearson_correlation method on the dataframe and columns to leave out
        corr_matrix = self.feature_engineering.pearson_correlation(self.df, self.columns_to_leave_out)

        # check that the returned value is a pandas DataFrame
        self.assertIsInstance(corr_matrix, pd.DataFrame)

        # check that the shape of the returned DataFrame match number of columns in dataframe minus one
        self.assertEqual(corr_matrix.shape, (self.df.shape[1]-len(self.columns_to_leave_out), self.df.shape[1]-len(self.columns_to_leave_out)))

        # check that all of the columns used in the correlation calculation are present in the returned corr_matrix
        self.assertTrue(all(col in corr_matrix.columns for col in self.df.columns if col not in self.columns_to_leave_out))

    def test_remove_constant_columns(self):
        """
        Test the remove_constant_columns method
        """
        # call the remove_constant_columns method on the dataframe and columns to leave out
        df = self.feature_engineering.remove_constant_columns(self.df, self.columns_to_leave_out)

        # check that the returned value is a pandas DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # check that the number of columns in the returned DataFrame match number of columns in dataframe minus the number of constant columns
        self.assertEqual(df.shape[1], self.df.shape[1]-len([col for col in self.df.columns if self.df[col].nunique() <= 1 and col not in self.columns_to_leave_out]))

        # check that all of the columns specified in columns_to_leave_out are present in the returned DataFrame
        self.assertTrue(all(col in df.columns for col in self.columns_to_leave_out))

    def test_remove_correlated_columns(self):
        """
        Test the remove_correlated_columns method
        """
        # call the remove_correlated_columns method on the dataframe, threshold and columns to leave out
        df_result = self.feature_engineering.remove_correlated_columns(self.df, self.threshold, self.columns_to_leave_out)

        # check that the returned value is a pandas DataFrame
        self.assertIsInstance(df_result, pd.DataFrame)

        # check that the columns that should be left out are still present in the returned dataframe
        self.assertTrue(all(col in df_result.columns for col in self.columns_to_leave_out))

        # check that the correlated columns have been removed and that the number of columns in the returned dataframe is less than the original dataframe
        corr_matrix = df_result.drop(self.columns_to_leave_out, axis=1).corr()
        correlated_columns = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    correlated_columns.add(colname)
        self.assertEqual(len(correlated_columns), 0)
        self.assertTrue(df_result.shape[1] < self.df.shape[1])

    def test_create_label_column(self):
        """
        Test the create_label_column method of the FeatureEngineering class
        """
        # call the create_label_column method
        result = self.feature_engineering.create_label_column(self.df)

        # check if the label column has been created with the correct values
        self.assertIn("label", result.columns)
        self.assertEqual(result["label"].tolist(), [1, 0, 1, 0, 1, 0])

        # check if the original gender column has been dropped
        self.assertNotIn("gender", result.columns)
