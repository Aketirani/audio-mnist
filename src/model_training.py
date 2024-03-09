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

    def set_params(self, model_param: dict):
        """
        Set model parameters

        :param model_param: dict, dictionary containing the parameters for the model
        """
        self.model.set_params(
            learning_rate=model_param["learning_rate"],
            max_depth=model_param["max_depth"],
            n_estimators=model_param["n_estimators"],
            objective=model_param["objective"],
            tree_method=model_param["tree_method"],
        )

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

    def _save_eval_metrics(
        self, file_path: str, file_name_results: str, result: object
    ) -> None:
        """
        Save the eval_metrics of a model in yaml format

        :param file_path: str, path where the eval_metrics should be saved
        :param result: object, the result returned by the fit method of a model
        """
        with open(os.path.join(file_path, file_name_results), "w") as f:
            json.dump(result.evals_result_, f)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        file_path: str,
        file_name_results: str,
        file_name_object: str,
    ) -> None:
        """
        Fit the model on training data

        :param X_train: np.ndarray, features for training
        :param y_train: np.ndarray, labels for training
        :param X_val: np.ndarray, features for validation
        :param y_val: np.ndarray, labels for validation
        :param file_path: str, path where the eval_metrics should be saved
        :param file_name_results: str, name of the model results file to be saved
        :param file_name_object: str, name of the model object file to be saved
        """
        result = self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=self._accuracy,
            verbose=0,
        )
        joblib.dump(self.model, os.path.join(file_path, file_name_object))
        self._save_eval_metrics(file_path, file_name_results, result)

    def _save_best_parameters(
        self, file_path: str, file_name: str, grid: object
    ) -> None:
        """
        Saves the best parameters of the best score of a model in a yaml file

        :param file_path: str, path where the best parameters and best score should be saved
        :param file_name: str, name of the file to save the best parameters
        :param grid: object, the grid returned by the fit method of a model
        """
        best_params = grid.best_params_
        with open(os.path.join(file_path, file_name), "w") as f:
            f.write("learning_rate: {}\n".format(best_params["learning_rate"]))
            f.write("max_depth: {}\n".format(best_params["max_depth"]))
            f.write("n_estimators: {}\n".format(best_params["n_estimators"]))
            f.write("objective: {}\n".format(best_params["objective"]))
            f.write("tree_method: {}\n".format(best_params["tree_method"]))

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        file_path: str,
        file_name: str,
        grid_params: dict,
    ) -> None:
        """
        Runs a grid search to tune the hyperparameters of the model

        :param X_train: np.ndarray, features for training
        :param y_train: np.ndarray, labels for training
        :param X_val: np.ndarray, features for validation
        :param y_val: np.ndarray, labels for validation
        :param file_path: str, path to save the best results
        :param file_name: str, name of the file to save the best results
        :param grid_params: str, dictionary containing the grid search parameters
        """
        grid = GridSearchCV(
            self.model, grid_params, cv=3, scoring="accuracy", n_jobs=-1
        )
        grid = grid.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )
        self._save_best_parameters(file_path, file_name, grid)

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
