import sys
import unittest

import pandas as pd

sys.path.append("../src")
from setup import Setup
from xgboost_model import XGBoostModel


class TestXGBoostModel(unittest.TestCase):
    """
    Test class for the XGBoostModel class
    """

    def setUp(self):
        """
        Set up the class with test fixtures.

        :param setup: class, create an instance of the Setup class
        :param train_df: pd.DataFrame, DataFrame for training data
        :param val_df: pd.DataFrame, DataFrame for validation data
        :param test_df: pd.DataFrame, DataFrame for test data
        :param xgboost_model: class, an instance of the XGBoostModel class
        :param model_param: dict, dictionary containing the parameters for the model
        :param log_data: dict, dictionary containing log data for the model
        :param expected_df: pd.DataFrame, expected result dataframe
        """
        self.setup = Setup(cfg_file="config.yaml")
        self.train_df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6],
                "col2": [1, 2, 3, 4, 5, 6],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        self.val_df = pd.DataFrame(
            {
                "col1": [7, 8, 9, 10, 11, 12],
                "col2": [7, 8, 9, 10, 11, 12],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        self.test_df = pd.DataFrame(
            {
                "col1": [13, 14, 15, 16, 17, 18],
                "col2": [13, 14, 15, 16, 17, 18],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        self.xgboost_model = XGBoostModel(self.train_df, self.val_df, self.test_df)
        self.model_param = {
            "learning_rate": 0.05,
            "max_depth": 5,
            "n_estimators": 500,
            "gamma": 0,
            "lambda": 1,
            "scale_pos_weight": 1,
            "min_child_weight": 1,
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

    def test_prepare_data(self):
        """
        Test the prepare_data method
        """
        # call the prepare_data method
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = self.xgboost_model.prepare_data()

        # Assert that X_train, X_val and X_test contain the expected features
        pd.testing.assert_frame_equal(X_train, self.xgboost_model.train_df.iloc[:, :-1])
        pd.testing.assert_frame_equal(X_val, self.xgboost_model.val_df.iloc[:, :-1])
        pd.testing.assert_frame_equal(X_test, self.xgboost_model.test_df.iloc[:, :-1])

        # Assert that y_train, y_val and y_test contain the expected labels
        pd.testing.assert_series_equal(y_train, self.xgboost_model.train_df.iloc[:, -1])
        pd.testing.assert_series_equal(y_val, self.xgboost_model.val_df.iloc[:, -1])
        pd.testing.assert_series_equal(y_test, self.xgboost_model.test_df.iloc[:, -1])

    def test_set_params(self):
        """
        Test the set_params method
        """
        # call the set_params method with the sample model_param dictionary
        self.xgboost_model.set_params(self.model_param)

        # check that the parameters of the model object have been set correctly
        self.assertEqual(
            self.xgboost_model.model.learning_rate, self.model_param["learning_rate"]
        )
        self.assertEqual(
            self.xgboost_model.model.max_depth, self.model_param["max_depth"]
        )
        self.assertEqual(
            self.xgboost_model.model.n_estimators, self.model_param["n_estimators"]
        )
        self.assertEqual(self.xgboost_model.model.gamma, self.model_param["gamma"])
        self.assertEqual(
            self.xgboost_model.model.reg_lambda, self.model_param["lambda"]
        )
        self.assertEqual(
            self.xgboost_model.model.scale_pos_weight,
            self.model_param["scale_pos_weight"],
        )
        self.assertEqual(
            self.xgboost_model.model.min_child_weight,
            self.model_param["min_child_weight"],
        )
        self.assertEqual(
            self.xgboost_model.model.objective, self.model_param["objective"]
        )
        self.assertEqual(
            self.xgboost_model.model.tree_method, self.model_param["tree_method"]
        )

    def test_create_log_df(self):
        """
        Test the create_log_df method
        """
        # get the resulting DataFrame from the method
        result_df = self.xgboost_model.create_log_df(self.log_data)

        # check if the resulting DataFrame match the expected one
        pd.testing.assert_frame_equal(result_df, self.expected_df)

    def test_evaluate_predictions(self):
        """
        Test the evaluate_predictions method
        """
        # calculate the result by calling the evaluate_predictions method
        result = self.xgboost_model.evaluate_predictions(
            self.val_df["label"].values, self.test_df["label"].values
        )

        # check that the result is equal to 1
        self.assertEqual(result, 1)
