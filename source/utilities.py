import pandas as pd
import os
import numpy as np
import scipy.io.wavfile as wavf
from typing import List, Tuple

class Utilities:
    def __init__(self):
        """
        Initialize the Utilities class
        """
        pass

    @staticmethod
    def read_audio(file: str) -> Tuple[int, np.ndarray]:
        """
        Read the audio data from the given file and return the sample rate and audio data
        
        :param file: str, path to the audio file
        :return: Tuple, containing the sample rate and audio data
        """
        return wavf.read(file)

    @staticmethod
    def create_dataframe(column_names: List[str]) -> pd.DataFrame:
        """
        Create an empty DataFrame with the given column names

        :param column_names: List[str], names of the columns for the DataFrame
        :return: pd.DataFrame, an empty DataFrame with the given column names
        """
        return pd.DataFrame(columns=column_names)

    @staticmethod
    def remove_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove a column from DataFrame

        :param df: pd.DataFrame, input DataFrame
        :param column: str, column name to remove
        :return: pd.DataFrame, DataFrame with column removed
        """
        if column in df.columns:
            df = df.drop([column], axis=1)
        else:
            print(f"{column} not found in DataFrame.")
        return df

    @staticmethod
    def save_df_to_csv(dataframe: pd.DataFrame, dst: str, file_name: str) -> None:
        """
        Save the given DataFrame to a CSV file

        :param dataframe: pd.DataFrame, DataFrame to be saved
        :param dst: str, destination where the CSV file will be saved
        :param file_name: str, name of the file to be saved
        """
        dataframe.to_csv(os.path.join(dst, file_name), index=False)

    @staticmethod
    def csv_to_dataframe(dst: str, file_name: str) -> pd.DataFrame:
        """
        Read the CSV file and return it as a Pandas DataFrame
        
        :param dst: str, destination where the CSV file is stored
        :param file_name: str, name of the file
        :return: pd.DataFrame, DataFrame created from the CSV file
        """
        return pd.read_csv(os.path.join(dst, file_name))

    @staticmethod
    def loop_progress(index:int, total:int):
        """
        This function takes in the current index, total number of iterations and sleep time 
        and displays the progress of the loop every iteration

        :param index: int, the current index of the loop
        :param total: int, total number of iterations in the loop
        """

        # calculate progress
        progress = (index) / (total)

        # print progress and elapsed time
        print(f'Progress: {progress:.2%}')
