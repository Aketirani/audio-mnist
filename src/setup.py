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
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_folder = os.path.join(script_dir, "../config")
            os.chdir(config_folder)
            with open(self.cfg_file, "r") as file:
                cfg_setup = yaml.safe_load(file)
        except:
            raise FileNotFoundError(f"{self.cfg_file} is not a valid config filepath!")
        return cfg_setup

    @staticmethod
    def read_file(filepath: str, filename: str) -> dict:
        """
        Read the file and return it as a dictionary

        :param filepath: str, path to the file to be read
        :param filepath: str, filename to be read
        :return: dict, containing the file data
        """
        path_file = os.path.join(filepath, filename)
        try:
            with open(path_file, "r") as file:
                file_data = yaml.safe_load(file)
        except:
            raise FileNotFoundError(f"{path_file} is not a valid filepath!")
        return file_data

    @staticmethod
    def extract_file_info(filepath: str) -> tuple:
        """
        Extracts information from the file path and splits it into parts

        :param filepath: str, path to the file
        :return: tuple, containing extracted information (dig, vp, rep)
        """
        dig, vp, rep = os.path.splitext(os.path.basename(filepath))[0].split("_")
        return dig, vp, rep

    def set_audio_path(self) -> str:
        """
        Get the path to the folder containing each participant's audio data

        :return: str, path to the audio folder
        """
        return os.path.join(self.read_config()["project"], "audio")

    def set_data_path(self) -> str:
        """
        Get the path to the folder containing data

        :return: str, path to the data folder
        """
        return os.path.join(self.read_config()["project"], "data")

    def set_img_path(self) -> str:
        """
        Get the path to the folder containing images for gui

        :return: str, path to the images folder
        """
        return os.path.join(self.read_config()["project"], "images")

    def set_param_path(self) -> str:
        """
        Get the path to the folder containing model parameters

        :return: str, path to the model parameters parameters folder
        """
        return os.path.join(self.read_config()["project"], "parameters")

    def set_plot_path(self) -> str:
        """
        Get the path to the folder containing plots

        :return: str, path to the plot folder
        """
        return os.path.join(self.read_config()["project"], "plots")

    def set_html_path(self) -> str:
        """
        Get the path to the folder containing html

        :return: str, path to the html folder
        """
        return os.path.join(self.read_config()["project"], "html")

    def set_result_path(self) -> str:
        """
        Get the path to the folder containing results

        :return: str, path to the plot folder
        """
        return os.path.join(self.read_config()["project"], "results")

    def set_test_path(self) -> str:
        """
        Get the path to the folder containing tests

        :return: str, path to the test folder
        """
        return os.path.join(self.read_config()["project"], "test")

    def set_txt_path(self) -> str:
        """
        Get the path to the folder containing txt for gui

        :return: str, path to the text folder
        """
        return os.path.join(self.read_config()["project"], "text")

    def set_model_path(self) -> str:
        """
        Get the path to the folder containing model parameters

        :return: str, path to the model parameters parameters folder
        """
        return os.path.join(self.read_config()["project"], "parameters")

    @staticmethod
    def loop_progress(index: int, total: int, show_every: int = 1):
        """
        Display the progress of the loop every iterations

        :param index: int, the current index of the loop
        :param total: int, total number of iterations in the loop
        :param show_every: int, optional, frequency of progress updates (default is 1)
        """
        if index % show_every == 0:
            progress = index / total
            print(f"Progress: {progress:.2%}")
