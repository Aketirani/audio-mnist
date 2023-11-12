import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append("../src")
from data_visualization import DataVisualization
from setup import Setup


class TestDataVisualization(unittest.TestCase):
    """
    Test class for the DataVisualization class
    """

    def setUp(self):
        """
        Set up the class with test fixtures

        :param setup: class, create an instance of the Setup class
        :param data_visualization: class, create an instance of the DataVisualization class
        :param data: np.ndarray, data array
        :param columns: np.ndarray, column names
        :param plot_name: str, plot name
        :param matrix: pd.DataFrame, data matrix
        """
        self.setup = Setup(cfg_file="config.yaml")
        self.data_visualization = DataVisualization(
            plot_path=self.setup.set_test_path()
        )
        self.data = np.array([0.1, 0.2, 0.3, 0.4])
        self.columns = ["col1", "col2", "col3", "col4"]
        self.plot_name = "test_plot.png"
        self.matrix = pd.DataFrame(
            [self.data, self.data, self.data, self.data], columns=self.columns
        )

    def test_plot_corr_matrix(self):
        """
        Test the plot_corr_matrix method
        """
        # plot the correlation matrix
        self.data_visualization.plot_corr_matrix(self.matrix, self.plot_name)

        # check if the plot has been saved at the specified location
        self.assertTrue(
            os.path.exists(os.path.join(self.setup.set_test_path(), self.plot_name))
        )

        # delete the plot file after the test
        os.remove(os.path.join(self.setup.set_test_path(), self.plot_name))

    def test_plot_loss(self):
        """
        Test the plot_loss method
        """
        # plot the loss
        self.data_visualization.plot_loss(
            self.data, self.data, self.data, self.plot_name
        )

        # check if the plot has been saved at the specified location
        self.assertTrue(
            os.path.exists(os.path.join(self.setup.set_test_path(), self.plot_name))
        )

        # delete the plot file after the test
        os.remove(os.path.join(self.setup.set_test_path(), self.plot_name))

    def test_plot_accuracy(self):
        """
        Test the plot_accuracy method
        """
        # plot the accuracy
        self.data_visualization.plot_accuracy(
            self.data, self.data, self.data, self.plot_name
        )

        # check if the plot has been saved at the specified location
        self.assertTrue(
            os.path.exists(os.path.join(self.setup.set_test_path(), self.plot_name))
        )

        # delete the plot file after the test
        os.remove(os.path.join(self.setup.set_test_path(), self.plot_name))

    def test_plot_column_dist(self):
        """
        Test the plot_column_dist method
        """
        # plot columns distribution
        self.data_visualization.plot_column_dist(self.matrix, self.plot_name)

        # check if the plot has been saved at the specified location
        self.assertTrue(
            os.path.exists(os.path.join(self.setup.set_test_path(), self.plot_name))
        )

        # delete the plot file after the test
        os.remove(os.path.join(self.setup.set_test_path(), self.plot_name))

    def test_plot_feature_importance(self):
        """
        Test the plot_feature_importance method
        """
        # plot feature importance
        self.data_visualization.plot_feature_importance(
            self.data, self.columns, self.plot_name
        )

        # check if the plot has been saved at the specified location
        self.assertTrue(
            os.path.exists(os.path.join(self.setup.set_test_path(), self.plot_name))
        )

        # delete the plot file after the test
        os.remove(os.path.join(self.setup.set_test_path(), self.plot_name))
