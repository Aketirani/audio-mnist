# Import python packages
import argparse
import glob
import os
import warnings
import pandas as pd

from src.data_preparation import DataPreparation
from src.data_split import DataSplit
from src.data_visualization import DataVisualization
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTraining
from src.model_predict import ModelPrediction
from src.setup import Setup
from src.utilities import Utilities

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

    def DataPrepare(self):
        """
        Data Preparation
        """
        # Read meta data file
        meta_data = UT.read_file(SU.set_audio_path(), self.config_file["meta_data"])

        # Create empty dataframe
        df = UT.create_dataframe(None, self.config_file["targets"])

        # Specify total number of folders in source path
        num_folders = len(next(os.walk(SU.set_audio_path()))[1]) + 1

        # Loop over audio recordings in the source path
        for i in range(1, num_folders):
            # Show progress
            UT.loop_progress(i, num_folders - 1)

            # Assign source temp
            src_temp = os.path.join(SU.set_audio_path(), f"{i:02d}")
            filepath_filename = sorted(glob.glob(os.path.join(src_temp, "*.wav")))

            # Loop over files in directory
            for file in filepath_filename:
                # Split file string
                dig, vp, rep = file.rstrip(".wav").split("/")[-1].split("_")

                # Read audio data
                fs, audio_data = UT.read_audio(file)

                # # Plot audio signal
                # audio_name = f"audio_{dig[-1]}_{vp}_{rep}.png"
                # DV.plot_audio(fs, audio_data, audio_name)

                # # Plot STFT of audio signal
                # stft_name = f"stft_{dig[-1]}_{vp}_{rep}.png"
                # DV.plot_stft(fs, audio_data, stft_name)

                # # Play audio signal
                # DV.play_audio(file)

                # Resample audio data
                audio_data = DP.resample_data(fs, audio_data)

                # Zero padding audio data
                audio_data = DP.zero_pad(audio_data)

                # FFT audio data
                fft_data = DP.fft_data(audio_data)

                # Feature creation
                features = DP.feature_creation(fft_data)

                # Normalize features
                n_features = DP.normalize_features(features)

                # Add gender and digit column
                features = UT.add_column_dict(
                    n_features,
                    self.config_file["targets"][0],
                    meta_data[vp][self.config_file["targets"][0]],
                )
                features = UT.add_column_dict(
                    n_features, self.config_file["targets"][1], dig[-1]
                )

                # Append new dict values to the DataFrame
                df = df.append(features, ignore_index=True)

        # Save processed data
        df.to_csv(
            os.path.join(
                SU.set_data_path(), self.config_file["data"]["data_prepared"]
            ),
            index=False,
        )

        # Show size of dataset
        print(f"Processed dataset, columns: {df.shape[1]} and rows: {df.shape[0]}")

    def FeatureEngineer(self):
        """
        Feature Engineering
        """
        # Load file into dataframe
        df = pd.read_csv(
            os.path.join(
                SU.set_data_path(), self.config_file["data"]["data_prepared"]
            )
        )

        # Remove digit column
        df = UT.remove_column(df, self.config_file["targets"][1])

        # Binarize target column where female is 0 and male is 1
        df = FE.binarize_column(df, self.config_file["targets"][0])

        # Plot column distribution
        DV.plot_column_dist(df, self.config_file["plots"]["column_distribution"])

        # Remove constant columns
        df = FE.remove_constant_columns(df, self.config_file["targets"][0])

        # Calculate correlation matrix
        corr_matrix = FE.pearson_correlation(df, self.config_file["targets"][0])

        # Plot correlation matrix
        DV.plot_corr_matrix(
            corr_matrix, self.config_file["plots"]["correlation_matrix"]
        )

        # Remove correlated columns
        df = FE.remove_correlated_columns(
            df, self.config_file["threshold"], self.config_file["targets"][0]
        )

        # Save engineered data
        df.to_csv(
            os.path.join(
                SU.set_data_path(), self.config_file["data"]["data_engineered"]
            ),
            index=False,
        )

    def DataSplit(self):
        """
        Data Splitting
        """
        # Load file into dataframe
        df = pd.read_csv(
            os.path.join(
                SU.set_data_path(), self.config_file["data"]["data_engineered"]
            )
        )

        # Split datasets
        self.train_df, self.val_df, self.test_df = DS.split(df, self.config_file["targets"][0])

        # Show size of datasets
        print(f"Training set, columns: {self.train_df.shape[1]} and rows: {self.train_df.shape[0]}")
        print(f"Validation set, columns: {self.val_df.shape[1]} and rows: {self.val_df.shape[0]}")
        print(f"Test set, columns: {self.test_df.shape[1]} and rows: {self.test_df.shape[0]}")

        # Show gender balance
        gender_count = UT.column_value_counts(df, self.config_file["targets"][0])
        print(f"Female audio recordings: {gender_count[0]}")
        print(f"Male audio recordings: {gender_count[1]}")

        # Prepare datasets
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = DS.prepare_data(
            self.train_df, self.val_df, self.test_df, self.config_file["targets"][0]
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
            self.config_file["results"]["model_param_best"],
            UT.read_file(
                SU.set_model_path(), self.config_file["parameters"]["model_dynamic"]
            ),
        )

    def ModelTrain(self):
        """
        Model Training
        """
        # Set model parameters
        MT.set_params(
            UT.read_file(
                SU.set_model_path(), self.config_file["parameters"]["model_static"]
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
            UT.read_file(
                SU.set_result_path(), self.config_file["results"]["model_results"]
            )
        )

        # Plot feature importance
        DV.plot_feature_importance(
            MT.feature_importance(),
            self.test_df.iloc[:, :-1].columns,
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
        df = UT.add_column_df(df, self.config_file["predicted"], y_pred)

        # Save predicted data
        df.to_csv(
            os.path.join(
                SU.set_data_path(), self.config_file["data"]["data_final"]
            ),
            index=False,
        )

        # Plot confusion matrix
        DV.plot_confusion_matrix(
            self.y_test,
            y_pred,
            self.config_file["labels"],
            self.config_file["plots"]["confusion_matrix"],
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
        "-d",
        "--data_prep",
        type=str,
        default="false",
        help="Prepare Data",
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
        default="true",
        help="Split Data",
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
        default="true",
        help="Model Training",
    )
    parser.add_argument(
        "-p",
        "--model_pred",
        type=str,
        default="true",
        help="Model Prediction",
    )
    args = parser.parse_args()

    # Initialize classes
    SU = Setup(args.cfg_file)
    UT = Utilities(SU.set_data_path())
    DP = DataPreparation()
    DV = DataVisualization(SU.set_plot_path())
    FE = FeatureEngineering()
    DS = DataSplit()
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
