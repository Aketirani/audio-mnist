import os

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavf
import yaml


class Utilities:
    """
    The Utilities class is used to read files, create dataframes, save csv files, and other basic functionalities
    """

    def __init__(self, data_path: str):
        """
        Initialize the Utilities class

        :param data_path: str, data path where the CSV file will be saved
        """
        self.data_path = data_path

    @staticmethod
    def read_audio(filepath: str) -> tuple:
        """
        Read the audio data from the given file and return the sample rate and audio data

        :param filepath: str, path to the audio file
        :return audio: tuple, containing the sample rate and audio data
        """
        try:
            # read the audio data from the file using the wavfile library
            audio = wavf.read(filepath)
        except:
            # raise an error if the filepath is not valid
            raise FileNotFoundError(f"{filepath} is not a valid filepath!")

        # return a tuple containing the sample rate and audio data
        return audio

    @staticmethod
    def read_file(filepath: str, filename: str) -> dict:
        """
        Read the file and return it as a dictionary

        :param filepath: str, path to the file to be read
        :param filepath: str, filename to be read
        :return: dict, containing the file data
        """
        # join filepath and filename
        path_file = os.path.join(filepath, filename)
        try:
            # open the file in read mode
            with open(path_file, "r") as file:
                # use yaml.safe_load() to parse the file and return it as a dictionary
                file_data = yaml.safe_load(file)
        except:
            # raise a FileNotFoundError if the filepath is not valid
            raise FileNotFoundError(f"{path_file} is not a valid filepath!")

        # return file data
        return file_data

    @staticmethod
    def create_dataframe(data: list, column_names: list) -> pd.DataFrame:
        """
        Create a DataFrame with the given column names and data (if provided)

        :param data: list[list], data to be used in the DataFrame, if None, empty dataframe will be created
        :param column_names: list, names of the columns for the DataFrame
        :return: pd.DataFrame, an DataFrame with the given column names and data (if provided)
        """
        # checking if data is None or not
        if data is None:
            # if data is None, create an empty dataframe with the given column names
            return pd.DataFrame(columns=column_names)
        else:
            # if data is provided, use it to create dataframe with the given column names
            return pd.DataFrame(data, columns=column_names)

    @staticmethod
    def df_shape(df: pd.DataFrame) -> tuple:
        """
        Find the shape of the given DataFrame

        :param df: pd.DataFrame, input DataFrame
        :return: tuple, containing the number of rows and columns in the DataFrame
        """
        # return the number of rows and columns in the dataframe
        return df.shape

    @staticmethod
    def column_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
        """
        Returns the value counts of a given column in a DataFrame

        :param df: pd.DataFrame, input DataFrame
        :param column: str, name of the column in the DataFrame
        :return: pd.Series, containing the value counts of the specified column
        """
        # return the value counts of the specified column in the dataframe
        return df[column].value_counts()

    @staticmethod
    def remove_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove a column from DataFrame

        :param df: pd.DataFrame, input DataFrame
        :param column: str, column name to remove
        :return: pd.DataFrame, DataFrame with column removed
        """
        # check if column exists in the dataframe
        if column in df.columns:
            # drop the column
            df = df.drop([column], axis=1)
        else:
            # if column does not exist, print message
            print(f"{column} not found in DataFrame")

        # return dataframe
        return df

    @staticmethod
    def add_column_dict(data: dict, column_name: str, value: str) -> dict:
        """
        Add column and value to the dictionary

        :param data: dict, input data
        :param column_name: str, column name to add
        :param value: str, value to add
        :return: dict, data with column added
        """
        # add the column to the dictionary
        data[column_name] = value

        # return dictionary
        return data

    @staticmethod
    def add_column_df(data: pd.DataFrame, column_name: str, value: str) -> pd.DataFrame:
        """
        Add column and value to the DataFrame

        :param data: DataFrame, input data
        :param column_name: str, column name to add
        :param value: str, value to add
        :return: DataFrame, data with column added
        """
        # add the column to the dataframe
        data[column_name] = value

        # return dataframe
        return data

    def save_df_to_csv(self, dataframe: pd.DataFrame, file_name: str) -> None:
        """
        Save the given DataFrame to a CSV file

        :param dataframe: pd.DataFrame, DataFrame to be saved
        :param file_name: str, name of the file to be saved
        """
        # save dataframe to csv file
        dataframe.to_csv(os.path.join(self.data_path, file_name), index=False)

    def csv_to_df(self, file_name: str) -> pd.DataFrame:
        """
        Read the CSV file and return it as a Pandas DataFrame

        :param file_name: str, name of the file
        :return: pd.DataFrame, DataFrame created from the CSV file
        """
        # save csv file to dataframe
        return pd.read_csv(os.path.join(self.data_path, file_name))

    @staticmethod
    def reshape_data(data_list: list) -> np.ndarray:
        """
        Reshape a 1D list

        :param data_list: list, 1D list to reshape
        :return: np.ndarray, reshaped 1D array
        """
        # convert list to numpy array
        data_list = np.array(data_list)

        # reshape numpy array
        return data_list.reshape(1, len(data_list))

    @staticmethod
    def loop_progress(index: int, total: int):
        """
        This function takes in the current index, total number of iterations and sleep time
        and displays the progress of the loop every iteration

        :param index: int, the current index of the loop
        :param total: int, total number of iterations in the loop
        """
        # calculate progress
        progress = (index) / (total)

        # print progress and elapsed time
        print(f"Progress: {progress:.2%}")
