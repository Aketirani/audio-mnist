from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

class XGBoostModel:
    def __init__(self, train_df, val_df, test_df):
        """
        Initialize the class with training, validation and test dataframes
        
        :param train_df: pandas.DataFrame, dataframe containing the training data
        :param val_df: pandas.DataFrame, dataframe containing the validation data
        :param test_df: pandas.DataFrame, dataframe containing the test data
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def prepare_data(self):
        """
        Prepare datasets for training, validation and test
        """
        # extract features for datasets
        X_train = self.train_df.iloc[:,:-1] 
        X_val = self.val_df.iloc[:,:-1]
        X_test = self.test_df.iloc[:,:-1]

        # extract labels for datasets
        y_train = self.train_df.iloc[:,-1]
        y_val = self.val_df.iloc[:,-1]
        y_test = self.test_df.iloc[:,-1]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def set_params(self, model_param: dict):
        """
        Set model parameters
        
        :param model_param: dict, dictionary containing the parameters for the model
        """
        # create an instance of the XGBClassifier
        self.model = XGBClassifier()

        # set the model parameters using the provided dictionary
        self.model.set_params(learning_rate=model_param["learning_rate"],
                     max_depth=model_param["max_depth"],
                     n_estimators=model_param["n_estimators"],
                     gamma=model_param["gamma"],
                     reg_lambda=model_param["lambda"],
                     scale_pos_weight=model_param["scale_pos_weight"],
                     min_child_weight=model_param["min_child_weight"],
                     objective=model_param["objective"],
                     tree_method=model_param["tree_method"],
                     verbosity=model_param["verbosity"])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Fit the model on training data
        
        :param X_train: numpy.ndarray, features for training
        :param y_train: numpy.ndarray, labels for training
        :param X_val: numpy.ndarray, features for validation
        :param y_val: numpy.ndarray, labels for validation
        """
        # define accuracy metric function
        def accuracy(preds, dtrain):
            labels = dtrain.get_label()
            return 'accuracy', accuracy_score(labels, np.round(preds))

        # fit the model on the training data and evaluate on validation data
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric=accuracy, verbose=True)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions for test data
        
        :param X_test: numpy.ndarray, features for test
        
        :return: numpy.ndarray, array containing the predicted labels for test data
        """
        # make predictions on the test data using the trained model
        y_pred = self.model.predict(X_test)
        # return the predicted labels
        return y_pred

    def evaluate_predictions(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate predictions and return accuracy
        
        :param y_test: numpy.ndarray, true labels for test data
        :param y_pred: numpy.ndarray, predicted labels for test data
        
        :return: float, accuracy score
        """
        # round the predictions to the nearest integer
        predictions = [round(value) for value in y_pred]

        # calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # return the accuracy
        return accuracy
