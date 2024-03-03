import os

import yaml


class Setup:
    """
    This class is used to read files, set up the configuration file and define paths
    """

    def __init__(self, cfg_file: str) -> None:
        """
        Initialize the class

        :param cfg_file: str, path to the configuration file
        """
        self.cfg_file = cfg_file

    def read_config(self) -> dict:
        """
        Read the config yaml file and return the data as a dictionary

        :return: dict, containing the configuration data
        """
        try:
            # change the directory to the configuration folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_folder = os.path.join(script_dir, "../config")
            os.chdir(config_folder)

            # open the configuration folder
            with open(self.cfg_file, "r") as file:
                # load the configuration file into a dictionary
                cfg_setup = yaml.safe_load(file)
        except:
            # raise an error if the filename is not valid
            raise FileNotFoundError(f"{self.cfg_file} is not a valid config filepath!")

        # return the configuration data
        return cfg_setup

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
    def extract_file_info(filepath: str) -> tuple:
        """
        Extracts information from the file path and splits it into parts

        :param filepath: str, path to the file
        :return: tuple, containing extracted information (dig, vp, rep)
        """
        # extract file name, remove extension, and split
        dig, vp, rep = os.path.splitext(os.path.basename(filepath))[0].split("_")

        # return parts
        return dig, vp, rep

    def set_audio_path(self) -> str:
        """
        Get the path to the folder containing each participant's audio data

        :return: str, path to the audio folder
        """
        # combine the project path and the audio folder path
        return os.path.join(self.read_config()["project_path"], "audio")

    def set_data_path(self) -> str:
        """
        Get the path to the folder containing data

        :return: str, path to the data folder
        """
        # combine the project path and the data folder path
        return os.path.join(self.read_config()["project_path"], "data")

    def set_param_path(self) -> str:
        """
        Get the path to the folder containing model parameters

        :return: str, path to the model parameters parameters folder
        """
        # combine the project path and the parameters folder path
        return os.path.join(self.read_config()["project_path"], "parameters")

    def set_plot_path(self) -> str:
        """
        Get the path to the folder containing plots

        :return: str, path to the plot folder
        """
        # combine the project path and the plots folder path
        return os.path.join(self.read_config()["project_path"], "plots")

    def set_result_path(self) -> str:
        """
        Get the path to the folder containing results

        :return: str, path to the plot folder
        """
        # combine the project path and the results folder path
        return os.path.join(self.read_config()["project_path"], "results")

    def set_test_path(self) -> str:
        """
        Get the path to the folder containing tests

        :return: str, path to the test folder
        """
        # combine the project path and the test folder path
        return os.path.join(self.read_config()["project_path"], "test")

    def set_model_path(self) -> str:
        """
        Get the path to the folder containing model parameters

        :return: str, path to the model parameters parameters folder
        """
        # combine the project path and the parameters folder path
        return os.path.join(self.read_config()["project_path"], "parameters")

    @staticmethod
    def loop_progress(index: int, total: int, show_every: int = 1):
        """
        This function takes in the current index, total number of iterations,
        and optional parameter 'show_every' to display the progress of the loop
        every 'show_every' iterations.

        :param index: int, the current index of the loop
        :param total: int, total number of iterations in the loop
        :param show_every: int, optional, frequency of progress updates (default is 1)
        """
        if index % show_every == 0:
            # calculate progress
            progress = index / total

            # print progress and elapsed time
            print(f"Progress: {progress:.2%}")
