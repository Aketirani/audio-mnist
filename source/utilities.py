import pandas as pd
import os
import numpy as np
import yaml
import scipy.io.wavfile as wavf

class Utilities:
    def __init__(self, dst: str):
        """
        Initialize the Utilities class

        :param dst: str, destination where the CSV file will be saved
        """
        self.dst = dst

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

    def read_file(self, filepath) -> dict:
        """
        Read the file and return the it as a dictionary
        
        :param filepath: str, path to the file to be read
        :return: dict, containing the file data
        """
        try:
            # open the file in read mode
            with open(filepath, 'r') as file:
                # use yaml.safe_load() to parse the file and return it as a dictionary
                file_data = yaml.safe_load(file)
        except:
            # raise a FileNotFoundError if the filepath is not valid
            raise FileNotFoundError(f"{filepath} is not a valid filepath!")

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
            print(f"{column} not found in DataFrame.")

        # return dataframe
        return df

    def save_df_to_csv(self, dataframe: pd.DataFrame, file_name: str) -> None:
        """
        Save the given DataFrame to a CSV file

        :param dataframe: pd.DataFrame, DataFrame to be saved
        :param file_name: str, name of the file to be saved
        """
        # save dataframe to csv file
        dataframe.to_csv(os.path.join(self.dst, file_name), index=False)

    def csv_to_df(self, file_name: str) -> pd.DataFrame:
        """
        Read the CSV file and return it as a Pandas DataFrame
        
        :param file_name: str, name of the file
        :return: pd.DataFrame, DataFrame created from the CSV file
        """
        # save csv file to dataframe
        return pd.read_csv(os.path.join(self.dst, file_name))

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
        return data_list.reshape(1,len(data_list))

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
