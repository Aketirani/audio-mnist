# Import python packages
import argparse
import glob
import os
import warnings

from src.data_processing import DataProcessing
from src.data_split import DataSplit
from src.data_visualization import DataVisualization
from src.feature_engineering import FeatureEngineering
from src.setup import Setup
from src.utilities import Utilities
from src.xgboost_model import XGBoostModel

# Ignore warnings
warnings.filterwarnings("ignore")


class AudioMNIST:
    def __init__(
        self,
        write_mode: bool,
        plot_mode: bool,
        play_mode: bool,
        print_mode: bool,
        train_mode: bool,
        tuning_mode: bool,
    ):
        """
        Initialize the class with the config file, and set up the paths and files

        :param config_file: dict, read the config file
        :param write_mode: bool, a flag indicating whether to write and save new data
        :param plot_mode: bool, a flag indicating whether to plot figures
        :param play_mode: bool, a flag indicating whether to play the audio signals
        :param print_mode: bool, a flag indicating whether to print
        :param train_mode, bool, a flag indicating wether to train model or load pre-trained model object
        :param tuning_mode: bool, a flag indicating whether to perform hyperparameter tuning
        """
        self.config_file = SU.cfg_setup
        self.write_mode = write_mode
        self.plot_mode = plot_mode
        self.play_mode = play_mode
        self.print_mode = print_mode
        self.train_mode = train_mode
        self.tuning_mode = tuning_mode

    def DataPreparation(self):
        """
        Prepares audio data for analysis
        """

        # Writes and saves new data
        if self.write_mode == True:
            # Read meta data file
            meta_data = UT.read_file(SU.meta_path)

            # Create empty dataframe
            df = UT.create_dataframe(None, self.config_file["targets"])

            # Specify total number of folders in source path
            num_folders = len(next(os.walk(SU.audio_path))[1]) + 1

            # Loop over audio recordings in the source path
            for i in range(1, num_folders):
                # Show progress
                if self.print_mode == True:
                    UT.loop_progress(i, num_folders - 1)

                # Assign source temp
                src_temp = os.path.join(SU.audio_path, f"{i:02d}")
                filepath_filename = sorted(glob.glob(os.path.join(src_temp, "*.wav")))

                # Loop over files in directory
                for file in filepath_filename:
                    # Split file string
                    dig, vp, rep = file.rstrip(".wav").split("/")[-1].split("_")

                    # Read audio data
                    fs, audio_data = UT.read_audio(file)

                    # Plot audio signal
                    if self.plot_mode == True:
                        audio_name = f"audio_{dig[-1]}_{vp}_{rep}.png"
                        DV.plot_audio(fs, audio_data, audio_name)

                    # Plot STFT of audio signal
                    if self.plot_mode == True:
                        stft_name = f"stft_{dig[-1]}_{vp}_{rep}.png"
                        DV.plot_stft(fs, audio_data, stft_name)

                    # Play audio signal
                    if self.play_mode == True:
                        DV.play_audio(file)

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

            # Save data to CSV
            UT.save_df_to_csv(df, file_name=self.config_file["data"]["data_prepared"])

    def DataEngineering(self):
        """
        Prepare final data for modelling
        """
        # Load CSV file into dataframe
        df = UT.csv_to_df(self.config_file["data"]["data_prepared"])

        if self.print_mode == True:
            # Show size of dataset
            print(
                f"Size of data set, columns: {UT.df_shape(df)[1]} and rows: {UT.df_shape(df)[0]}"
            )

        # Remove digit column
        df = UT.remove_column(df, self.config_file["targets"][1])

        # Binarize target column where female is 0 and male is 1
        df = FE.binarize_column(df, self.config_file["targets"][0])

        if self.plot_mode == True:
            # Plot column distribution
            DV.plot_column_dist(df, self.config_file["plots"]["column_distribution"])

        # Remove constant columns
        df = FE.remove_constant_columns(df, self.config_file["targets"][0])

        # Calculate correlation matrix
        corr_matrix = FE.pearson_correlation(df, self.config_file["targets"][0])

        if self.plot_mode == True:
            # Plot correlation matrix
            DV.plot_corr_matrix(
                corr_matrix, self.config_file["plots"]["correlation_matrix"]
            )

        # Remove correlated columns
        df = FE.remove_correlated_columns(
            df, self.config_file["threshold"], self.config_file["targets"][0]
        )

        # Save data to CSV
        UT.save_df_to_csv(df, self.config_file["data"]["data_engineered"])

    def Modelling(self):
        """
        Hyperparameter tuning, model training, prediction and evaluation
        """
        # Load CSV file into dataframe
        df = UT.csv_to_df(self.config_file["data"]["data_engineered"])

        # Split datasets
        train_df, val_df, test_df = DS.split(df, self.config_file["targets"][0])

        if self.print_mode == True:
            # Show size of datasets
            train_size = UT.df_shape(train_df)
            val_size = UT.df_shape(val_df)
            test_size = UT.df_shape(test_df)
            print(
                f"Size of training set, columns: {train_size[1]} and rows: {train_size[0]}"
            )
            print(
                f"Size of validation set, columns: {val_size[1]} and rows: {val_size[0]}"
            )
            print(f"Size of test set, columns: {test_size[1]} and rows: {test_size[0]}")

            # Show gender balance
            gender_count = UT.column_value_counts(df, self.config_file["targets"][0])
            print(f"Number of female audio recordings: {gender_count[0]}")
            print(f"Number of male audio recordings: {gender_count[1]}")

        # Prepare datasets
        X_train, y_train, X_val, y_val, X_test, y_test = DS.prepare_data(
            train_df, val_df, test_df, self.config_file["targets"][0]
        )

        # Initialize
        XM = XGBoostModel(X_train, y_train, X_val, y_val, X_test, y_test)

        # Hyperparameters tuning
        if self.tuning_mode == True:
            XM.grid_search(
                SU.res_path,
                self.config_file["results"]["model_param_best"],
                UT.read_file(SU.hyperparam_path),
            )

        # Set model parameters
        XM.set_params(UT.read_file(SU.param_path))

        # Model training
        if self.train_mode == True:
            # Set model parameters
            XM.set_params(UT.read_file(SU.param_path))

            # Train model
            XM.fit(
                SU.res_path,
                self.config_file["results"]["model_results"],
                self.config_file["results"]["model_object"],
            )
        else:
            # Load the pre-trained model object
            XM.model = XM.load_model(SU.res_path, self.config_file["results"]["model_object"])

        # Feature importance
        feature_importance = XM.feature_importance()

        # Plot feature importance
        if self.plot_mode == True:
            DV.plot_feature_importance(
                feature_importance,
                test_df.iloc[:, :-1].columns,
                self.config_file["plots"]["feature_importance"],
            )

        # Make predictions
        y_pred = XM.predict()

        # Create final dataframe from test set and reset index
        df = test_df.reset_index(drop=True)

        # Add predicted values column to final dataframe
        df = UT.add_column_df(df, self.config_file["predicted"], y_pred)

        # Save data to CSV
        UT.save_df_to_csv(df, self.config_file["data"]["data_final"])

        # Plot confusion matrix
        if self.plot_mode == True:
            DV.plot_confusion_matrix(
                y_test,
                y_pred,
                self.config_file["labels"],
                self.config_file["plots"]["confusion_matrix"],
            )

        # Evaluate model
        accuracy = XM.evaluate_predictions(y_pred)
        if self.print_mode == True:
            print("Model Accuracy: %.2f%%" % (accuracy * 100))

        # Read model results
        model_results = UT.read_file(
            os.path.join(SU.res_path, self.config_file["results"]["model_results"])
        )

        # Load results into pandas dataframe
        df = XM.create_log_df(model_results)

        if self.plot_mode == True:
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
        "-w",
        "--write_mode",
        type=str,
        default=False,
        help="Write New Data",
    )
    parser.add_argument(
        "-o",
        "--plot_mode",
        type=str,
        default=False,
        help="Plot Figures",
    )
    parser.add_argument(
        "-y",
        "--play_mode",
        type=str,
        default=False,
        help="Play Audio",
    )
    parser.add_argument(
        "-i",
        "--print_mode",
        type=str,
        default=True,
        help="Print Statements",
    )
    parser.add_argument(
        "-t",
        "--train_mode",
        type=bool,
        default=False,
        help="Model Training",
    )
    parser.add_argument(
        "-u",
        "--tuning_mode",
        type=bool,
        default=False,
        help="Hyperparameter Tuning",
    )
    args = parser.parse_args()

    # Initialize classes
    SU = Setup(args.cfg_file)
    UT = Utilities(SU.data_path)
    DP = DataProcessing()
    DV = DataVisualization(SU.plot_path)
    FE = FeatureEngineering()
    DS = DataSplit()

    # Call main class
    AM = AudioMNIST(
        args.write_mode,
        args.plot_mode,
        args.play_mode,
        args.print_mode,
        args.train_mode,
        args.tuning_mode,
    )

    # Prepare raw audio data for analysis
    AM.DataPreparation()

    # Prepare processed data for modelling
    AM.DataEngineering()

    # Train model and make predictions
    AM.Modelling()
