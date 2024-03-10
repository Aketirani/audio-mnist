import os
import unittest

import pandas as pd

from src.model_training import ModelTraining
from src.setup import Setup


class TestModelTraining(unittest.TestCase):
    """
    Test class for the ModelTraining class
    """

    def setUp(self):
        """
        Set up the class with test fixtures.

        :param setup: class, create an instance of the Setup class
        :param model_train: class, an instance of the ModelTraining class
        :param model_param_fit: dict, dictionary containing the parameters for the model
        :param model_param_grid: dict, dictionary containing the hyperparameters for the model
        :param log_data: dict, dictionary containing log data for the model
        :param expected_df: pd.DataFrame, expected result dataframe
        :param model_name: str, model object name
        """
        self.setup = Setup(cfg_file="config.yaml")
        self.model_train = ModelTraining()
        self.model_param_fit = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 100,
            "objective": "binary:logistic",
            "tree_method": "hist",
        }
        self.model_param_grid = {
            "learning_rate": [0.1, 0.2],
            "max_depth": [3, 5],
            "n_estimators": [100, 200],
            "objective": "binary:logistic",
            "tree_method": "hist",
        }
        self.log_data = {
            "validation_0": {"logloss": [0.5, 0.4, 0.3], "accuracy": [0.3, 0.4, 0.5]},
            "validation_1": {"logloss": [0.4, 0.3, 0.2], "accuracy": [0.2, 0.3, 0.4]},
        }
        self.expected_df = pd.DataFrame(
            {
                "iteration": [0, 1, 2],
                "train_loss": [0.5, 0.4, 0.3],
                "train_acc": [0.3, 0.4, 0.5],
                "val_loss": [0.4, 0.3, 0.2],
                "val_acc": [0.2, 0.3, 0.4],
            }
        )
        self.model_name = "test_model.pkl"

    def test_set_params_fit(self):
        self.model_train.set_params_fit(self.model_param_fit)
        self.assertEqual(
            self.model_train.model.learning_rate, self.model_param_fit["learning_rate"]
        )
        self.assertEqual(
            self.model_train.model.max_depth, self.model_param_fit["max_depth"]
        )
        self.assertEqual(
            self.model_train.model.n_estimators, self.model_param_fit["n_estimators"]
        )
        self.assertEqual(
            self.model_train.model.objective, self.model_param_fit["objective"]
        )
        self.assertEqual(
            self.model_train.model.tree_method, self.model_param_fit["tree_method"]
        )

    def test_set_params_grid(self):
        self.model_train.set_params_grid(self.model_param_grid)
        self.assertEqual(self.model_train.grid_params, self.model_param_grid)

    def test_save_model_object(self):
        self.model_train.save_model_object(
            self.setup.set_result_path(), self.model_name
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.setup.set_result_path(), self.model_name))
        )
        os.remove(os.path.join(self.setup.set_result_path(), self.model_name))

    def test_create_log_df(self):
        result_df = self.model_train.create_log_df(self.log_data)
        pd.testing.assert_frame_equal(result_df, self.expected_df)
