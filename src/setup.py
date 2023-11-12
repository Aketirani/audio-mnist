import os

import yaml


class Setup:
    """
    This class is used to read files, set up the configuration file and define paths
    """

    def __init__(self, cfg_file: str) -> None:
        """
        Initialize the class
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

    def set_audio_path(self) -> str:
        """
        Get the path to the folder containing each participant's audio data

        :return: str, path to the audio folder
        """
        # combine the project path and the audio folder path
        return os.path.join(self.cfg_setup["project_path"], "audio")

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
