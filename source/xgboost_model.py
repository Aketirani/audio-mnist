import xgboost as xgb

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
        
    def train(self, params):
        """
        Train the XGBoost model on the training data
        
        :param params: dict, a dictionary containing the parameters for the XGBoost model
        """
        # convert dataframes to DMatrix
        dtrain = xgb.DMatrix(self.train_df.drop('label', axis=1), label=self.train_df['label'])
        dval = xgb.DMatrix(self.val_df.drop('label', axis=1), label=self.val_df['label'])

        # train the model
        self.model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "val")], early_stopping_rounds=10)

    def predict(self):
        """
        Make predictions on the test data
        
        :return: numpy.ndarray, array of predictions made by the model
        """
        dtest = xgb.DMatrix(self.test_df.drop('label', axis=1), label=self.test_df['label'])
        return self.model.predict(dtest)
