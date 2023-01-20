import pandas as pd
import os
import numpy as np
import scipy.io.wavfile as wavf
from typing import List, Tuple

class Utilities:
    def __init__(self):
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
    def save_df_to_csv(dataframe: pd.DataFrame, dst: str, file_name: str) -> None:
        """
        Save the given DataFrame to a CSV file

        :param dataframe: pd.DataFrame, DataFrame to be saved
        :param dst: str, destination where the CSV file will be saved
        :param file_name: str, name of the file to be saved
        """
        dataframe.to_csv(os.path.join(dst, file_name), index=False)

    @staticmethod
    def csv_to_dataframe(file: str) -> pd.DataFrame:
        """
        Read the CSV file and return it as a Pandas DataFrame
        
        :param file: str, path to the CSV file
        :return: pd.DataFrame, DataFrame created from the CSV file
        """
        return pd.read_csv(file)
