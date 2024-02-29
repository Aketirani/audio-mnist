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
        # Read meta data file
        meta_data = SU.read_file(SU.set_audio_path(), self.config_file["meta_data"])

        # Create empty dataframe
        df = pd.DataFrame()

        # Specify total number of folders in source path
        num_folders = len(next(os.walk(SU.set_audio_path()))[1]) + 1

        # Loop over audio recordings in the source path
        for i in range(1, num_folders):
            # Show progress
            SU.loop_progress(i, num_folders - 1, 6)

            # Loop over files in directory
            audio_file = sorted(
                glob.glob(
                    os.path.join(os.path.join(SU.set_audio_path(), f"{i:02d}"), "*.wav")
                )
            )
            for file in audio_file:
                # Split file string
                dig, vp, rep = file.rstrip(".wav").split("/")[-1].split("_")

                # Read audio data
                fs, audio_data = DP.read_audio(file)

                # Resample audio data
                audio_data = DP.resample_data(fs, audio_data)

                # Zero padding audio data
                audio_data = DP.zero_pad(audio_data)

                # Calculate time-domain features
                time_domain_features = DP.feature_creation_time_domain(audio_data)

                # FFT audio data
                fft_data = DP.fft_data(audio_data)

                # Calculate frequency-domain features
                features = DP.feature_creation_frequency_domain(fft_data)

                # Merge features
                features.update(time_domain_features)

                # Normalize features
                n_features = DP.normalize_features(features)

                # Add gender and digit column
                features = DP.add_column_dict(
                    n_features,
                    self.config_file["target"],
                    meta_data[vp][self.config_file["target"]],
                )

                # Append new dict values to the DataFrame
                df = df.append(features, ignore_index=True)

        # Plot audio signal
        audio_name = f"audio_{dig[-1]}_{vp}_{rep}.png"
        DV.plot_audio(fs, audio_data, audio_name, 1)

        # Plot STFT of audio signal
        stft_name = f"stft_{dig[-1]}_{vp}_{rep}.png"
        DV.plot_stft(fs, audio_data, stft_name, 1)

        # Play audio signal
        DV.play_audio(file, 1)

        # Save prepared data to csv
        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"]),
            index=False,
        )

        # Save prepared data to PostgreSQL
        PM.write_csv_to_table(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"]),
            self.pgs_file["table"]["prepared"],
        )

        # Show gender balance
        gender_count = DP.column_value_counts(df, self.config_file["target"])
        print(f"Female audio recordings: {gender_count[0]}")
        print(f"Male audio recordings: {gender_count[1]}")

        # Show size of dataset
        print(f"Prepared dataset, columns: {df.shape[1]} and rows: {df.shape[0]}")

    def FeatureEngineer(self):
        """
        Feature Engineering
        """
        # Load file into dataframe
        df = pd.read_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"])
        )

        # Remove constant columns
        df = FE.remove_constant_columns(df, self.config_file["target"])

        # Catogarize target column where female is 0 and male is 1
        df = FE.categorize_column_values(df, self.config_file["target"])

        # Plot column distribution
        DV.plot_column_dist(
            df,
            self.config_file["plots"]["column_distribution"],
            self.config_file["target"],
        )

        # Calculate correlation matrix
        corr_matrix = FE.pearson_correlation(df, self.config_file["target"])

        # Plot correlation matrix
        DV.plot_corr_matrix(
            corr_matrix, self.config_file["plots"]["correlation_matrix"]
        )

        # Remove correlated columns
        df = FE.remove_correlated_columns(
            df,
            self.config_file["thresholds"]["correlation"],
            self.config_file["target"],
        )

        # Save engineered data to csv
        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"]),
            index=False,
        )

        # Drop table in PostgreSQL
        PM.drop_table(self.pgs_file["table"]["engineered"])

        # Create table from csv in PostgreSQL
        PM.create_table_from_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"]),
            self.pgs_file["table"]["engineered"],
        )

        # Save engineered data to PostgreSQL
        PM.write_df_to_table(df, self.pgs_file["table"]["engineered"])

    def DataSplit(self):
        """
        Data Splitting
        """
        # Load file into dataframe
        df = pd.read_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"])
        )

        # Split datasets
        self.train_df, self.val_df, self.test_df = DS.split(
            df, self.config_file["target"]
        )

        # Show size of datasets
        print(
            f"Training set, columns: {self.train_df.shape[1]} and rows: {self.train_df.shape[0]}"
        )
        print(
            f"Validation set, columns: {self.val_df.shape[1]} and rows: {self.val_df.shape[0]}"
        )
        print(
            f"Test set, columns: {self.test_df.shape[1]} and rows: {self.test_df.shape[0]}"
        )

        # Prepare datasets
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
        # Hyperparameters tuning through grid search
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
        # Set model parameters
        MT.set_params(
            SU.read_file(
                SU.set_model_path(), self.config_file["parameters"]["model_parameters"]
            )
        )

        # Train model
        MT.fit(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            SU.set_result_path(),
            self.config_file["results"]["model_results"],
            self.config_file["results"]["model_object"],
        )

        # Load results into pandas dataframe
        df = MT.create_log_df(
            SU.read_file(
                SU.set_result_path(), self.config_file["results"]["model_results"]
            )
        )

        # Plot feature importance
        DV.plot_feature_importance(
            MT.model,
            self.config_file["plots"]["feature_importance"],
        )

        # Plot training and validation accuracy and loss
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
        # Load the pre-trained model object
        MT.model = MP.load_model(
            SU.set_result_path(), self.config_file["results"]["model_object"]
        )

        # Make predictions
        y_pred = MP.predict(MT.model, self.test_df.iloc[:, :-1])

        # Create final dataframe from test set and reset index
        df = self.test_df.reset_index(drop=True)

        # Add predicted values column to final dataframe
        df = DP.add_column_df(df, self.config_file["predicted"], y_pred)

        # Save predicted data to csv
        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["predicted"]),
            index=False,
        )

        # Save predicted data to PostgreSQL
        PM.write_csv_to_table(
            os.path.join(SU.set_data_path(), self.config_file["data"]["predicted"]),
            self.pgs_file["table"]["predicted"],
        )

        # Plot confusion matrix
        DV.plot_confusion_matrix(
            self.y_test,
            y_pred,
            self.config_file["labels"],
            self.config_file["plots"]["confusion_matrix"],
        )

        # Plot Shapley summary
        DV.plot_shapley_summary(
            MT.model,
            self.test_df.iloc[:, :-1],
            self.config_file["plots"]["shapley_summary"],
        )

        # Evaluate model
        MP.evaluate_predictions(self.y_test, y_pred)


if __name__ == "__main__":
    # Add arguments
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
        default="false",
        help="Data Preparation",
    )
    parser.add_argument(
        "-f",
        "--feat_eng",
        type=str,
        default="true",
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
    args = parser.parse_args()

    # Initialize classes
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
