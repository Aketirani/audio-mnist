import pandas as pd


class FeatureEngineering:
    """
    The FeatureEngineering class is used to prepare final data for modelling
    """

    def __init__(self):
        """
        Initialize the FeatureEngineering class
        """
        pass

    @staticmethod
    def pearson_correlation(
        df: pd.DataFrame, columns_to_leave_out: list
    ) -> pd.DataFrame:
        """
        Calculate the Pearson correlation coefficient on a DataFrame, excluding the specified columns

        :param df: pd.DataFrame, DataFrame on which correlation has to be calculated
        :param columns_to_leave_out: list, list of column names to exclude while calculating correlation
        :return: pd.DataFrame, DataFrame containing correlation coefficients
        """
        columns_to_use = [col for col in df.columns if col not in columns_to_leave_out]
        return df[columns_to_use].corr(method="pearson")

    def remove_constant_columns(
        self, df: pd.DataFrame, columns_to_leave_out: list
    ) -> pd.DataFrame:
        """
        Remove columns with constant values from the DataFrame, except for the ones specified in columns_to_leave_out

        :param df: pd.DataFrame, input DataFrame
        :param columns_to_leave_out: list, columns to exclude from removal
        :return: pd.DataFrame, DataFrame with constant value columns removed
        """
        constant_columns = [
            col
            for col in df.columns
            if df[col].nunique() <= 1 and col not in columns_to_leave_out
        ]
        return df.drop(constant_columns, axis=1)

    @staticmethod
    def remove_correlated_columns(
        df: pd.DataFrame, threshold: float, columns_to_leave_out: list
    ) -> pd.DataFrame:
        """
        Remove correlated columns from the DataFrame that have a correlation above the given threshold

        :param df: pd.DataFrame, DataFrame to remove correlated columns from
        :param threshold: float, correlation threshold above which columns will be removed
        :param columns_to_leave_out: list, column names to leave out of the correlation calculation but keep in the final output
        :return: pd.DataFrame, DataFrame with correlated columns removed
        """
        df_original = df.copy()
        corr_matrix = df.drop(columns_to_leave_out, axis=1).corr()
        correlated_columns = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    correlated_columns.add(colname)
        df = df.drop(correlated_columns, axis=1)
        for col in [columns_to_leave_out]:
            if col not in df.columns:
                df[col] = df_original[col]
        return df

    @staticmethod
    def categorize_column_values(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Categorize the specified column values to numerical values

        :param df: pd.DataFrame, input DataFrame
        :param column_name: str, column name to be categorized
        :return: pd.DataFrame, DataFrame with the categorized column
        """
        df[column_name] = df[column_name].map({"female": 0, "male": 1})
        return df
