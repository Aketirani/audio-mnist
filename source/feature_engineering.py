import pandas as pd
import numpy as np
from typing import List

class FeatureEngineering:
    def __init__(self):
        """
        Initialize the FeatureEngineering class
        """
        pass

    @staticmethod
    def pearson_correlation(df: pd.DataFrame, columns_to_leave_out: List[str]=[]) -> pd.DataFrame:
        """
        Calculate the Pearson correlation coefficient on a DataFrame, excluding the specified columns
        
        :param df: pd.DataFrame, DataFrame on which correlation has to be calculated
        :param columns_to_leave_out: List[str], list of column names to exclude while calculating correlation
        :return: pd.DataFrame, DataFrame containing correlation coefficients
        """
        
        # drop columns with only one unique value
        df = df.loc[:, df.nunique() > 1]
        
        # list of columns to use for correlation calculation
        columns_to_use = [col for col in df.columns if col not in columns_to_leave_out]
        
        # calculate correlation matrix for the columns
        return df[columns_to_use].corr(method='pearson')

    def remove_constant_columns(self, df: pd.DataFrame, columns_to_leave_out: List[str]) -> pd.DataFrame:
        """
        Remove columns with constant values from the DataFrame, except for the ones specified in 'columns_to_leave_out'

        :param df: pd.DataFrame, input DataFrame
        :param columns_to_leave_out: List[str], columns to exclude from removal
        :return: pd.DataFrame, DataFrame with constant value columns removed
        """
        # Get the columns with constant values
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1 and col not in columns_to_leave_out]

        # Remove the constant value columns
        df = df.drop(constant_columns, axis=1)

        return df

    @staticmethod
    def remove_correlated_columns(df: pd.DataFrame, threshold: float, columns_to_leave_out: List[str]) -> pd.DataFrame:
        """
        Remove correlated columns from the DataFrame that have a correlation above the given threshold
        
        :param df: pd.DataFrame, DataFrame to remove correlated columns from
        :param threshold: float, correlation threshold above which columns will be removed
        :param columns_to_leave_out: List[str], column names to leave out of the correlation calculation but keep in the final output
        :return: pd.DataFrame, DataFrame with correlated columns removed
        """

        # Store a copy of the original DataFrame
        df_original = df.copy()

        # Calculate correlation matrix
        corr_matrix = df.drop(columns_to_leave_out, axis=1).corr()
        
        # Identify correlated columns
        correlated_columns = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    correlated_columns.add(colname)
        
        # Drop correlated columns
        df = df.drop(correlated_columns, axis=1)
        
        # Add back the columns that were left out
        for col in columns_to_leave_out:
            if col not in df.columns:
                df[col] = df_original[col]
                
        return df
