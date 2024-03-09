import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class ModelTraining:
    """
    This class is used to do hyperparameter tuning and model training
    """

    def __init__(self):
        """
        Initialize the class
        """
        self.model = XGBClassifier()

    def set_params_fit(self, model_param: dict):
        """
        Set model parameters for model training

        :param model_param: dict, dictionary containing the model parameters
        """
        self.model.set_params(**model_param)

    def _accuracy(self, preds: np.ndarray, dtrain: object) -> tuple:
        """
        Calculates the accuracy of predictions made by a model

        :param preds: np.array, an array of predictions made by the model
        :param dtrain: object, the training data that is used to evaluate the accuracy of the model
        :return: tuple, a tuple containing the string 'accuracy' and the calculated accuracy score
        """
        labels = dtrain.get_label()
        accuracy = accuracy_score(labels, np.round(preds))
        return "accuracy", accuracy

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Fit the model on training data

        :param X_train: np.ndarray, features for training
        :param y_train: np.ndarray, labels for training
        :param X_val: np.ndarray, features for validation
        :param y_val: np.ndarray, labels for validation
        """
        self.result = self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=self._accuracy,
            verbose=0,
        )

    def save_model_object(self, file_path: str, file_name: object) -> None:
        """
        Save the object of a model

        :param file_path: str, path where the eval_metrics should be saved
        :param file_name: str, name of the model object file to be saved
        """
        joblib.dump(self.model, os.path.join(file_path, file_name))

    def save_eval_metrics(self, file_path: str, file_name_results: str) -> None:
        """
        Save the eval metrics of a model

        :param file_path: str, path where the eval_metrics should be saved
        :param file_name_results: str, name of the model results file to be saved
        """
        with open(os.path.join(file_path, file_name_results), "w") as f:
            json.dump(self.result.evals_result_, f)

    def set_params_grid(self, grid_params: dict):
        """
        Set grid search hyperparameters for model tuning

        :param grid_params: dict, dictionary containing the model hyperparameters
        """
        self.grid_params = grid_params

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Runs a grid search to tune the hyperparameters of the model

        :param X_train: np.ndarray, features for training
        :param y_train: np.ndarray, labels for training
        :param X_val: np.ndarray, features for validation
        :param y_val: np.ndarray, labels for validation
        :param file_path: str, path to save the best results
        :param file_name: str, name of the file to save the best results
        """
        grid = GridSearchCV(
            self.model, self.grid_params, cv=3, scoring="accuracy", n_jobs=-1
        )
        self.grid = grid.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )

    def save_best_parameters(self, file_path: str, file_name: str) -> None:
        """
        Saves the best parameters of the best score of a model in a yaml file

        :param file_path: str, path where the best parameters should be saved
        :param file_name: str, name of the file to save the best parameters
        """
        best_params = self.grid.best_params_
        with open(os.path.join(file_path, file_name), "w") as f:
            f.write("learning_rate: {}\n".format(best_params["learning_rate"]))
            f.write("max_depth: {}\n".format(best_params["max_depth"]))
            f.write("n_estimators: {}\n".format(best_params["n_estimators"]))
            f.write("objective: {}\n".format(best_params["objective"]))
            f.write("tree_method: {}\n".format(best_params["tree_method"]))

    def feature_importance(self) -> np.ndarray:
        """
        Calculates the feature importance of the model
        """
        return self.model.feature_importances_

    @staticmethod
    def create_log_df(log_data: dict) -> pd.DataFrame:
        """
        Create a DataFrame from log data

        :param log_data: dict, log data
        :return: pd.DataFrame, DataFrame containing log data
        """
        train_loss = log_data["validation_0"]["logloss"]
        val_loss = log_data["validation_1"]["logloss"]
        train_acc = log_data["validation_0"]["accuracy"]
        val_acc = log_data["validation_1"]["accuracy"]
        iteration = list(range(0, len(train_loss)))
        columns = ["iteration", "train_loss", "train_acc", "val_loss", "val_acc"]
        return pd.DataFrame(
            list(zip(iteration, train_loss, train_acc, val_loss, val_acc)),
            columns=columns,
        )
