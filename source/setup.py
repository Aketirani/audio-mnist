import os
import yaml


class Setup:
    """
    Setup all basic inputs for the project pipeline
    """

    def __init__(self, cfg_file: str) -> None:
        """
        Initialize the class with the config file, and set up the paths and files

        :param cfg_file: str, name to the config file
        """
        self.cfg_file = cfg_file
        self.cfg_setup = self.read_config()
        self.source_path = self.set_source_path()
        self.meta_path = self.set_meta_data_path()
        self.destination_path = self.set_destination_path()
        self.plot_path = self.set_plot_path()
        self.result_path = self.set_result_path()
        self.model_folder_path = self.set_model_folder_path()
        self.param_path = self.set_model_param_path()
        self.hyperparam_path = self.set_model_hyperparam_path()

    def read_config(self) -> dict:
        """
        Read the config yaml file and return the data as a dictionary

        :return: dict, containing the configuration data
        """
        try:
            # change the directory to the configuration folder
            os.chdir(os.path.join(os.getcwd(), "config"))

            # open the configuration folder
            with open(self.cfg_file, 'r') as file:
                # load the configuration file into a dictionary
                cfg_setup = yaml.safe_load(file)
        except:
            # raise an error if the filename is not valid
            raise FileNotFoundError(
                f"{self.cfg_file} is not a valid config filepath!")

        # return the configuration data
        return cfg_setup

    def set_source_path(self) -> str:
        """
        Get the path to the folder containing each participant's data

        :return: str, path to the data folder
        """
        # Combine the project path and the data folder path
        return os.path.join(self.cfg_setup['project_path'], "data")

    def set_meta_data_path(self) -> str:
        """
        Get the path to the file containing meta data information

        :return: str, path to the meta data file
        """
        # Combine the project path and the data folder path to get the meta data file path
        return os.path.join(self.cfg_setup['project_path'], "data", "audioMNIST_meta.txt")

    def set_destination_path(self) -> str:
        """
        Get the path to the folder containing each participant's preprocessed data

        :return: str, path to the preprocessed data_pre folder
        """
        # Combine the project path and the data_pre folder path
        return os.path.join(self.cfg_setup['project_path'], "data_pre")

    def set_plot_path(self) -> str:
        """
        Get the path to the folder containing plots

        :return: str, path to the plot folder
        """
        # Combine the project path and the plots folder path
        return os.path.join(self.cfg_setup['project_path'], "plots")

    def set_result_path(self) -> str:
        """
        Get the path to the folder containing results

        :return: str, path to the plot folder
        """
        # Combine the project path and the results folder path
        return os.path.join(self.cfg_setup['project_path'], "results")

    def set_model_folder_path(self) -> str:
        """
        Get the path to the folder containing model parameters

        :return: str, path to the model parameters parameters folder
        """
        # Combine the project path and the parameters folder path
        return os.path.join(self.cfg_setup['project_path'], "parameters")

    def set_model_param_path(self) -> str:
        """
        Get the path to the file containing model parameters information

        :return: str, path to the model parameters file
        """
        # Combine the project path and the parameters folder path to get the model parameters file path
        return os.path.join(self.cfg_setup['project_path'], "parameters", "model_parameters.yaml")

    def set_model_hyperparam_path(self) -> str:
        """
        Get the path to the file containing model hyperparameters information

        :return: str, path to the model hyperparameters file
        """
        # Combine the project path and the parameters folder path to get the model hyperparameters file path
        return os.path.join(self.cfg_setup['project_path'], "parameters", "model_hyperparameters.yaml")
