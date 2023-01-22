import os
import yaml
from typing import (Dict, List, Tuple, TypeVar)

# Define a generic type variable for any type that can be used in the code
ANY = TypeVar("ANY", Dict, List, Tuple, str, int, float)

class Setup:
    """
    Setup all basic inputs for the project pipeline
    """
    def __init__(self, cfg_filepath: str) -> None:
        """
        Initialize the class with the config file path and set up the other paths and meta data
        
        :param cfg_filepath: str, path to the config file
        """
        self.cfg_filepath = cfg_filepath
        self.cfg_setup = self.read_config()
        self.source_meta_path = self.set_meta_data_path()
        self.source_path = self.set_source_path()
        self.destination_path = self.set_destination_path()
        self.plot_path = self.set_plot_path()
        self.result_path = self.set_result_path()
        self.meta_data = self.read_meta_data()

    def read_config(self) -> Dict[str, ANY]:
        """
        Read the config yaml file and return the data as a dictionary
        
        :return: Dict, containing the config data
        """

        try:
            with open(self.cfg_filepath, 'r') as file:
                cfg_setup = yaml.safe_load(file)
        except:
            raise FileNotFoundError(f"{self.cfg_filepath} is not a valid config filepath!")
        return cfg_setup

    def set_source_path(self) -> str:
        """
        Get the path to the folder containing each participant's data
        
        :return: str, path to the data folder
        """
        return os.path.join(self.cfg_setup['project_path'], "data")

    def set_meta_data_path(self) -> str:
        """
        Get the path to the file containing meta data information
        
        :return: str, path to the meta data file
        """
        return os.path.join(self.cfg_setup['project_path'], "data", "audioMNIST_meta.txt")

    def set_destination_path(self) -> str:
        """
        Get the path to the folder containing each participant's preprocessed data
        
        :return: str, path to the preprocessed data_pre folder
        """
        return os.path.join(self.cfg_setup['project_path'], "data_pre")

    def set_plot_path(self) -> str:
        """
        Get the path to the folder containing plots
        
        :return: str, path to the plot folder
        """
        return os.path.join(self.cfg_setup['project_path'], "plots")

    def set_result_path(self) -> str:
        """
        Get the path to the folder containing results
        
        :return: str, path to the plot folder
        """
        return os.path.join(self.cfg_setup['project_path'], "results")

    def read_meta_data(self) -> Dict[str, ANY]:
        """
        Read the meta data yaml file and return the data as a dictionary
        
        :return: Dict, containing the meta data
        """

        try:
            with open(self.source_meta_path, 'r') as file:
                meta_data = yaml.safe_load(file)
        except:
            raise FileNotFoundError(f"{self.source_meta_path} is not a valid meta data filepath!")
        return meta_data
