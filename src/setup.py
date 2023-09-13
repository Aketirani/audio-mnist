import os

import yaml


class Setup:
    """
    The Setup class is used to set up paths
    """

    def __init__(self, cfg_file: str) -> None:
        """
        Initialize the class with the config file, and set up the paths and files

        :param cfg_file: str, name to the config file
        :param cfg_setup: dict, read the config file
        :param audio_path: str, path to the audio folder
        :param data_path: str, path to the data folder
        :param plot_path: str, path to the plots folder
        :param res_path: str, path to the results folder
        :param test_path: str, path to the test folder
        :param model_path: str, path to the model parameters folder
        """
        self.cfg_file = cfg_file
        self.cfg_setup = self.read_config()
        self.audio_path = self.set_audio_path()
        self.data_path = self.set_data_path()
        self.plot_path = self.set_plot_path()
        self.res_path = self.set_result_path()
        self.test_path = self.set_test_path()
        self.model_path = self.set_model_path()

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

    def set_audio_path(self) -> str:
        """
        Get the path to the folder containing each participant's audio data

        :return: str, path to the audio folder
        """
        # combine the project path and the audio folder path
        return os.path.join(self.cfg_setup["project_path"], "audio")

    def set_data_path(self) -> str:
        """
        Get the path to the folder containing each participant's processed data

        :return: str, path to the processed data folder
        """
        # combine the project path and the data folder path
        return os.path.join(self.cfg_setup["project_path"], "data")

    def set_plot_path(self) -> str:
        """
        Get the path to the folder containing plots

        :return: str, path to the plot folder
        """
        # combine the project path and the plots folder path
        return os.path.join(self.cfg_setup["project_path"], "plots")

    def set_result_path(self) -> str:
        """
        Get the path to the folder containing results

        :return: str, path to the plot folder
        """
        # combine the project path and the results folder path
        return os.path.join(self.cfg_setup["project_path"], "results")

    def set_test_path(self) -> str:
        """
        Get the path to the folder containing tests

        :return: str, path to the test folder
        """
        # combine the project path and the test folder path
        return os.path.join(self.cfg_setup["project_path"], "test")

    def set_model_path(self) -> str:
        """
        Get the path to the folder containing model parameters

        :return: str, path to the model parameters parameters folder
        """
        # combine the project path and the parameters folder path
        return os.path.join(self.cfg_setup["project_path"], "parameters")
