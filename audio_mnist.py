import argparse
import glob
import os
import warnings

import pandas as pd

from src.data_preparation import DataPreparation
from src.data_splitting import DataSplitting
from src.data_visualization import DataVisualization
from src.feature_engineering import FeatureEngineering
from src.model_prediction import ModelPrediction
from src.model_training import ModelTraining
from src.postgres import PostgresManager
from src.setup import Setup

warnings.filterwarnings("ignore")


class AudioMNIST:
    """
    This class is used to run the main pipeline
    """

    def __init__(self):
        """
        Initialize the class
        """
        self.config_file = SU.read_config()
        self.pgs_file = PM.read_config()

    def DataPrepare(self):
        """
        Data Preparation
        """
        # read meta data file
        meta_data = SU.read_file(SU.set_audio_path(), self.config_file["meta_data"])

        # create empty dataframe
        df = pd.DataFrame()

        # specify total number of folders in source path
        num_folders = len(next(os.walk(SU.set_audio_path()))[1]) + 1

        # loop over audio recordings in the source path
        for i in range(1, num_folders):
            # show progress
            SU.loop_progress(i, num_folders - 1, 6)

            # loop over files in directory
            audio_file = sorted(
                glob.glob(os.path.join(SU.set_audio_path(), f"{i:02d}", "*.wav"))
            )

            for file in audio_file:
                # split file string
                dig, vp, rep = SU.extract_file_info(file)

                # read audio data
                fs, audio_data = DP.read_audio(file)

                # resample audio data
                audio_data = DP.resample_data(fs, audio_data)

                # zero padding audio data
                audio_data = DP.zero_pad(audio_data)

                # calculate time-domain features
                time_domain_features = DP.feature_creation_time_domain(audio_data)

                # FFT audio data
                fft_data = DP.fft_data(audio_data)

                # calculate frequency-domain features
                features = DP.feature_creation_frequency_domain(fft_data)

                # merge features
                features.update(time_domain_features)

                # normalize features
                n_features = DP.normalize_features(features)

                # add target
                features = DP.add_column_dict(
                    n_features,
                    self.config_file["target"],
                    meta_data[vp][self.config_file["target"]],
                )

                # append new dict values to the DataFrame
                df = pd.concat(
                    [df, pd.DataFrame(features, index=[0])], ignore_index=True
                )

                # plot audio signal
                audio_name = self.config_file["plots"]["audio"].format(dig[-1], vp, rep)
                DV.plot_audio(fs, audio_data, audio_name, 1)

                # plot STFT of audio signal
                stft_name = self.config_file["plots"]["stft"].format(dig[-1], vp, rep)
                DV.plot_stft(fs, audio_data, stft_name, 1)

                # play audio signal
                DV.play_audio(file, 0)

        # save prepared data to csv
        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"]),
            index=False,
        )

        # show gender balance
        gender_count = DP.column_value_counts(df, self.config_file["target"])
        print(f"Female audio recordings: {gender_count[0]}")
        print(f"Male audio recordings: {gender_count[1]}")

        # show size of dataset
        print(f"Prepared dataset, columns: {df.shape[1]} and rows: {df.shape[0]}")

    def FeatureEngineer(self):
        """
        Feature Engineering
        """
        # load file into dataframe
        df = pd.read_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"])
        )

        # remove constant columns
        df = FE.remove_constant_columns(df, self.config_file["target"])

        # catogarize target column where female is 0 and male is 1
        df = FE.categorize_column_values(df, self.config_file["target"])

        # plot column distribution
        DV.plot_column_dist(
            df,
            self.config_file["plots"]["column_distribution"],
            self.config_file["target"],
        )

        # calculate correlation matrix
        corr_matrix = FE.pearson_correlation(df, self.config_file["target"])

        # plot correlation matrix
        DV.plot_corr_matrix(
            corr_matrix, self.config_file["plots"]["correlation_matrix"]
        )

        # remove correlated columns
        df = FE.remove_correlated_columns(
            df,
            self.config_file["thresholds"]["correlation"],
            self.config_file["target"],
        )

        # save engineered data to csv
        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"]),
            index=False,
        )

    def DataSplit(self):
        """
        Data Splitting
        """
        # load file into dataframe
        df = pd.read_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"])
        )

        # split datasets
        self.train_df, self.val_df, self.test_df = DS.split(
            df, self.config_file["target"]
        )

        # show size of datasets
        print(
            f"Training set, columns: {self.train_df.shape[1]} and rows: {self.train_df.shape[0]}"
        )
        print(
            f"Validation set, columns: {self.val_df.shape[1]} and rows: {self.val_df.shape[0]}"
        )
        print(
            f"Test set, columns: {self.test_df.shape[1]} and rows: {self.test_df.shape[0]}"
        )

        # prepare datasets
        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = DS.prepare_data(
            self.train_df, self.val_df, self.test_df, self.config_file["target"]
        )

    def ModelTune(self):
        """
        Model Hyperparameter Tuning
        """
        # hyperparameters tuning through grid search
        MT.grid_search(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            SU.set_result_path(),
            self.config_file["results"]["model_best_param"],
            SU.read_file(
                SU.set_model_path(),
                self.config_file["parameters"]["model_hyperparameters"],
            ),
        )

    def ModelTrain(self):
        """
        Model Training
        """
        # set model parameters
        MT.set_params(
            SU.read_file(
                SU.set_model_path(), self.config_file["parameters"]["model_parameters"]
            )
        )

        # train model
        MT.fit(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            SU.set_result_path(),
            self.config_file["results"]["model_results"],
            self.config_file["results"]["model_object"],
        )

        # load results into pandas dataframe
        df = MT.create_log_df(
            SU.read_file(
                SU.set_result_path(), self.config_file["results"]["model_results"]
            )
        )

        # plot feature importance
        DV.plot_feature_importance(
            MT.model,
            self.config_file["plots"]["feature_importance"],
        )

        # plot training and validation loss and accuracy
        DV.plot_loss(
            df["iteration"],
            df["train_loss"],
            df["val_loss"],
            self.config_file["plots"]["model_loss"],
        )
        DV.plot_accuracy(
            df["iteration"],
            df["train_acc"],
            df["val_acc"],
            self.config_file["plots"]["model_accuracy"],
        )

    def ModelPredict(self):
        """
        Model Prediction And Evaluation
        """
        # load the pre-trained model object
        MT.model = MP.load_model(
            SU.set_result_path(), self.config_file["results"]["model_object"]
        )

        # make predictions
        y_pred = MP.predict(MT.model, self.X_test)

        # create final dataframe from test set and reset index
        df = self.test_df.reset_index(drop=True)

        # add predicted values column to final dataframe
        df = DP.add_column_df(df, self.config_file["predicted"], y_pred)

        # save predicted data to csv
        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["predicted"]),
            index=False,
        )

        # plot confusion matrix
        DV.plot_confusion_matrix(
            self.y_test,
            y_pred,
            self.config_file["labels"],
            self.config_file["plots"]["confusion_matrix"],
        )

        # plot shapley summary
        DV.plot_shapley_summary(
            MT.model,
            self.X_test,
            self.config_file["plots"]["shapley_summary"],
        )

        # evaluate model
        MP.evaluate_predictions(self.y_test, y_pred)

    def DataSQL(self):
        """
        Data To PostgreSQL
        """
        # drop tables
        PM.drop_table(self.pgs_file["table"]["prepared"])
        PM.drop_table(self.pgs_file["table"]["engineered"])
        PM.drop_table(self.pgs_file["table"]["predicted"])

        # create table schemas
        PM.create_table_from_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"]),
            self.pgs_file["table"]["prepared"],
            self.config_file["target"],
        )
        PM.create_table_from_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"]),
            self.pgs_file["table"]["engineered"],
        )
        PM.create_table_from_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["predicted"]),
            self.pgs_file["table"]["predicted"],
        )

        # save data
        PM.write_csv_to_table(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"]),
            self.pgs_file["table"]["prepared"],
        )
        PM.write_csv_to_table(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"]),
            self.pgs_file["table"]["engineered"],
        )
        PM.write_csv_to_table(
            os.path.join(SU.set_data_path(), self.config_file["data"]["predicted"]),
            self.pgs_file["table"]["predicted"],
        )


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cfg_file",
        type=str,
        default="config.yaml",
        help="Configuration File",
    )
    parser.add_argument(
        "-y",
        "--pgs_file",
        type=str,
        default="postgres.yaml",
        help="Postgres File",
    )
    parser.add_argument(
        "-d",
        "--data_prep",
        type=str,
        default="true",
        help="Data Preparation",
    )
    parser.add_argument(
        "-f",
        "--feat_eng",
        type=str,
        default="false",
        help="Feature Engineering",
    )
    parser.add_argument(
        "-s",
        "--data_split",
        type=str,
        default="false",
        help="Data Splitting",
    )
    parser.add_argument(
        "-u",
        "--model_tune",
        type=str,
        default="false",
        help="Model Tuning",
    )
    parser.add_argument(
        "-t",
        "--model_train",
        type=str,
        default="false",
        help="Model Training",
    )
    parser.add_argument(
        "-p",
        "--model_pred",
        type=str,
        default="false",
        help="Model Prediction",
    )
    parser.add_argument(
        "-q",
        "--data_sql",
        type=str,
        default="false",
        help="Data To PostgreSQL",
    )
    args = parser.parse_args()

    # initialize classes
    SU = Setup(args.cfg_file)
    PM = PostgresManager(args.pgs_file)
    DP = DataPreparation()
    DV = DataVisualization(SU.set_plot_path())
    FE = FeatureEngineering()
    DS = DataSplitting()
    MT = ModelTraining()
    MP = ModelPrediction()
    AM = AudioMNIST()

    data_split_required = (
        args.model_train == "true"
        or args.model_tune == "true"
        or args.model_pred == "true"
    )

    if args.data_prep == "true":
        AM.DataPrepare()

    if args.feat_eng == "true":
        AM.FeatureEngineer()

    if args.data_split == "true" or data_split_required:
        AM.DataSplit()

    if args.model_tune == "true":
        AM.ModelTune()

    if args.model_train == "true":
        AM.ModelTrain()

    if args.model_pred == "true":
        AM.ModelPredict()

    if args.data_sql == "true":
        AM.DataSQL()
