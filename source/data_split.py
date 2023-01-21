import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplit:
    """
    The DataSplit class is used to split a dataframe into training, validation, and test sets.
    """
    def __init__(self, test_size: float = 0.1, val_size: float = 0.1):
        """
        Initialize the DataSplit class

        :param test_size: float, proportion of data to be used for the test set (default=0.1)
        :param val_size: float, proportion of data to be used for the validation set (default=0.1)
        """
        self.test_size = test_size
        self.val_size = val_size
        
    def split(self, dataframe: pd.DataFrame, target_column: str) -> tuple:
        """
        Split the dataframe into training, validation, and test sets.
        
        :param dataframe: pd.DataFrame, dataframe to be split
        :param target_column: str, name of the column to stratify on
        :return: tuple, containing the training, validation, and test dataframes in that order
        """
        # Split the dataframe into a training set and a validation+test set
        train_df, val_test_df = train_test_split(dataframe, test_size=self.val_size + self.test_size, stratify=dataframe[target_column])
        
        # Split the validation+test set further into a validation set and a test set
        val_df, test_df = train_test_split(val_test_df, test_size=self.test_size/(self.val_size+self.test_size), stratify=val_test_df[target_column])
        
        return train_df, val_df, test_df
