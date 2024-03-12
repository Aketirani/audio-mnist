import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitting:
    """
    This class is used to split a dataframe into training, validation, and test sets
    """

    def __init__(self):
        """
        Initialize the class
        """
        pass

    def split(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        val_size: float,
        test_size: float,
    ) -> tuple:
        """
        Split the dataframe into training, validation, and test sets

        :param dataframe: pd.DataFrame, dataframe to be split
        :param target_column: str, name of the column to stratify on
        :param val_size: float, proportion of data to be used for the validation set
        :param test_size: float, proportion of data to be used for the test set
        :return: tuple, containing the training, validation, and test dataframes in that order
        """
        train_df, val_test_df = train_test_split(
            dataframe,
            test_size=val_size + test_size,
            stratify=dataframe[target_column],
        )
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=test_size / (val_size + test_size),
            stratify=val_test_df[target_column],
        )
        return train_df, val_df, test_df

    @staticmethod
    def prepare_data(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
    ) -> tuple:
        """
        Prepare datasets for training, validation, and test

        :param train_df: pd.DataFrame, dataframe containing the training data
        :param val_df: pd.DataFrame, dataframe containing the validation data
        :param test_df: pd.DataFrame, dataframe containing the test data
        :param target_column: str, the name of the target column
        :return: tuple, containing X_train, y_train, X_val, y_val, X_test, y_test
        """
        X_train = train_df.drop(columns=[target_column])
        X_val = val_df.drop(columns=[target_column])
        X_test = test_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        y_val = val_df[target_column]
        y_test = test_df[target_column]
        return X_train, y_train, X_val, y_val, X_test, y_test
