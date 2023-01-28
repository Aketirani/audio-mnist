import pandas as pd
import os
import numpy as np
import yaml
import scipy.io.wavfile as wavf
from typing import List, Tuple, Dict

class Utilities:
    def __init__(self, dst: str):
        """
        Initialize the Utilities class

        :param dst: str, destination where the CSV file will be saved
        """
        self.dst = dst

    @staticmethod
    def read_audio(filepath: str) -> Tuple[int, np.ndarray]:
        """
        Read the audio data from the given file and return the sample rate and audio data
        
        :param filepath: str, path to the audio file
        :return audio: Tuple, containing the sample rate and audio data
        """
        try:
            audio = wavf.read(filepath)
        except:
            raise FileNotFoundError(f"{filepath} is not a valid filepath!")
        return audio

    def read_file(self, filepath) -> Dict:
        """
        Read the file and return the it as a dictionary
        
        :return: Dict, containing the file data
        """

        try:
            with open(filepath, 'r') as file:
                file_data = yaml.safe_load(file)
        except:
            raise FileNotFoundError(f"{filepath} is not a valid filepath!")
        return file_data

    @staticmethod
    def create_dataframe(data: List[List], column_names: List[str]) -> pd.DataFrame:
        """
        Create a DataFrame with the given column names and data (if provided)

        :param data: List[List[float]], data to be used in the DataFrame, if None, empty dataframe will be created
        :param column_names: List[str], names of the columns for the DataFrame
        :return: pd.DataFrame, an DataFrame with the given column names and data (if provided)
        """
        if data is None:
            return pd.DataFrame(columns=column_names)
        else:
            return pd.DataFrame(data, columns=column_names)

    @staticmethod
    def df_shape(df: pd.DataFrame) -> Tuple[int,int]:
        """
        Find the shape of the given DataFrame

        :param df: pd.DataFrame, input DataFrame
        :return: Tuple, containing the number of rows and columns in the DataFrame
        """
        return df.shape

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

    def save_df_to_csv(self, dataframe: pd.DataFrame, file_name: str) -> None:
        """
        Save the given DataFrame to a CSV file

        :param dataframe: pd.DataFrame, DataFrame to be saved
        :param file_name: str, name of the file to be saved
        """
        dataframe.to_csv(os.path.join(self.dst, file_name), index=False)

    def csv_to_df(self, file_name: str) -> pd.DataFrame:
        """
        Read the CSV file and return it as a Pandas DataFrame
        
        :param file_name: str, name of the file
        :return: pd.DataFrame, DataFrame created from the CSV file
        """
        return pd.read_csv(os.path.join(self.dst, file_name))

    @staticmethod
    def reshape_data(list: List) -> np.ndarray:
        """
        Reshape a 1D list

        :param list: List, 1D list to reshape
        :return: np.ndarray, reshaped 1D array
        """
        
        # convert list to numpy array
        list = np.array(list)

        # reshape numpy array
        return list.reshape(1,len(list))

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
