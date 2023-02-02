# Import python packages
import glob
import os
import warnings
import pandas as pd
from src.data_processing import DataProcessing
from src.data_split import DataSplit
from src.data_visualization import DataVisualization
from src.feature_engineering import FeatureEngineering
from src.setup import Setup
from src.utilities import Utilities
from src.xgboost_model import XGBoostModel

# Ignore warnings
warnings.filterwarnings("ignore")

# Initialize class
SU = Setup(cfg_file="config.yaml")

# Get the paths
source_path = SU.source_path
meta_path = SU.meta_path
destination_path = SU.destination_path
plot_path = SU.plot_path
result_path = SU.result_path
model_folder_path = SU.model_folder_path
model_param_path = SU.param_path
model_hyperparam_path = SU.hyperparam_path

# Initialize class
UT = Utilities(destination_path)

# Read files
meta_data = UT.read_file(meta_path)
model_param = UT.read_file(model_param_path)
model_hyperparam = UT.read_file(model_hyperparam_path)

# Initialize classes
DP = DataProcessing(target_sr=8000)
DV = DataVisualization(plot_path)
FE = FeatureEngineering()
DS = DataSplit()
DS = DataSplit(test_size=0.1, val_size=0.1)

def DataPreparation(plot_mode: bool, play_mode: bool, print_mode: bool, test_mode: bool):
    """
    Prepares audio data for analysis

    :param plot_mode: bool, a flag indicating whether to plot figures
    :param play_mode: bool, a flag indicating whether to play the audio signals
    :param print_mode: bool, a flag indicating whether to print
    :param test_mode: bool, a flag indicating whether to run the function in test mode
    :return: None
    """
    # Create empty dataframe
    df = UT.create_dataframe(data=None, column_names=["gender", "digit"])

    # Specify total number of folders in source path
    if (test_mode == True):
        num_folders = 2
    elif (test_mode == False):
        num_folders = len(next(os.walk(source_path))[1])+1

    # Loop over audio recordings in the source path
    for i in range(1, num_folders):
        # Show progress
        if (print_mode == True):
            UT.loop_progress(i, num_folders-1)

        # Assign source temp
        src_temp = os.path.join(source_path, f"{i:02d}")
        filepath_filename = sorted(glob.glob(os.path.join(src_temp, "*.wav")))

        # Loop over files in directory
        for file in filepath_filename:
            # Split file string
            dig, vp, rep = file.rstrip(".wav").split("/")[-1].split("_")

            # Read audio data
            fs, audio_data = UT.read_audio(file)

            # Plot audio signal
            if (plot_mode == True):
                audio_name = f"audio_{dig[-1]}_{vp}_{rep}.png"
                DV.plot_audio(fs, audio_data, audio_name)

            # Plot STFT of audio signal
            if (plot_mode == True):
                stft_name = f"stft_{dig[-1]}_{vp}_{rep}.png"
                DV.plot_stft(fs, audio_data, stft_name)

            # Play audio signal
            if (play_mode == True):
                DV.play_audio(file)

            # Resample audio data
            audio_data = DP.resample_data(fs, audio_data)

            # Zero padding audio data
            audio_data = DP.zero_pad(audio_data)

            # FFT audio data
            fft_data = DP.fft_data(audio_data)

            # Apply bandpass filter
            bp_data = DP.bandpass_filter(fft_data, low_threshold=100, high_threshold=250)

            # Feature creation
            features = DP.feature_creation(fft_data)

            # Normalize features
            n_features = DP.normalize_features(features)

            # Add gender and digit label
            features = UT.add_column(n_features, "gender", meta_data[vp]["gender"])
            features = UT.add_column(n_features, "digit", dig[-1])

            # Append new dict values to the DataFrame
            df = df.append(features, ignore_index=True)

            # Break
            if (test_mode == True):
                break

    # Save data to CSV
    if (test_mode == False):
        UT.save_df_to_csv(df, csv_name="features_data.csv")

def DataFinal(plot_mode: bool, print_mode: bool):
    """
    Prepare final data for modelling

    :param plot_mode: bool, a flag indicating whether to plot figures
    :param print_mode: bool, a flag indicating whether to print
    :return: None
    """
    # Load CSV file into dataframe
    df = UT.csv_to_df(file_name="features_data.csv")

    if (print_mode == True):
        # Show size of dataset
        print(f"Size of data set, columns: {UT.df_shape(df)[1]} and rows: {UT.df_shape(df)[0]}")

    # Remove digit column
    df = UT.remove_column(df, "digit")

    # Create label column where 'female' is 0 and 'male' is 1
    df = FE.create_label_column(df)

    if (plot_mode == True):
        # Plot column distribution
        DV.column_distribution(df, plot_name="column_distribution.png")

    # Remove constant columns
    df = FE.remove_constant_columns(df, columns_to_leave_out=["label"])

    # Calculate correlation matrix
    corr_matrix = FE.pearson_correlation(df, columns_to_leave_out=["label"])

    if (plot_mode == True):
        # Plot correlation matrix
        DV.plot_corr_matrix(corr_matrix, plot_name="correlation_matrix.png")

    # Remove correlated columns
    df = FE.remove_correlated_columns(df, threshold=0.95, columns_to_leave_out=["label"])

    # Save data to CSV
    UT.save_df_to_csv(df, file_name="final_data.csv")

def DataSplit(print_mode: bool) -> pd.DataFrame:
    """
    Split the final data into training, validation, and test sets

    :param print_mode: bool, a flag indicating whether to print
    :return: pd.DataFrame, dataframe containing train_df, val_df, test_df
    """
    # Load CSV file into dataframe
    df = UT.csv_to_df(file_name="final_data.csv")

    # Split datasets
    train_df, val_df, test_df = DS.split(df, "label")

    if (print_mode == True):
        # Show size of datasets
        train_size = UT.df_shape(train_df)
        val_size = UT.df_shape(val_df)
        test_size = UT.df_shape(test_df)
        print(f"Size of training set, columns: {train_size[1]} and rows: {train_size[0]}")
        print(f"Size of validation set, columns: {val_size[1]} and rows: {val_size[0]}")
        print(f"Size of validation set, columns: {test_size[1]} and rows: {test_size[0]}")

        # Show gender balance
        gender_count = UT.column_value_counts(df, "label")
        print(f"Number of female audio recordings: {gender_count[0]}")
        print(f"Number of male audio recordings: {gender_count[1]}")

    return train_df, val_df, test_df

def Modelling(plot_mode: bool, print_mode: bool, hyperparam_mode: bool):
    """
    Hyperparameter tuning, model training, prediction and evaluation

    :param plot_mode: bool, a flag indicating whether to plot figures
    :param print_mode: bool, a flag indicating whether to print
    :param model_hyperparam: bool, a flag indicating whether to perform hyperparameter tuning
    :return: None
    """
    # Prepare datasets
    X_train, y_train, X_val, y_val, X_test, y_test = XM.prepare_data()

    # Hyperparameters tuning
    if (hyperparam_mode == True):
        XM.grid_search(X_train, y_train, X_val, y_val, result_path, file_name="best_modeL_param.yaml", grid_params=model_hyperparam)

    # Set model parameters
    XM.set_params(model_param)

    # Train model
    XM.fit(X_train, y_train, X_val, y_val, result_path, file_name="model_results.yaml")

    # Feature importance
    feature_importance = XM.feature_importance()

    # Plot feature importance
    if (plot_mode == True):
        DV.plot_feature_importance(feature_importance, test_df.iloc[:,:-1].columns, plot_name="feature_importance.png")

    # Make predictions
    y_pred = XM.predict(X_test)

    # Evaluate model
    accuracy = XM.evaluate_predictions(y_test, y_pred)
    if (print_mode == True):
        print("Model Accuracy: %.2f%%" % (accuracy*100))

    # Read model results
    model_results = UT.read_file(os.path.join(result_path,"model_results.yaml"))

    # Load results into pandas dataframe
    df = XM.create_log_df(model_results)

    if (plot_mode == True):
        # Plot training and validation accuracy and loss
        DV.plot_loss(df["iteration"], df["train_loss"], df["val_loss"], plot_name="model_loss.png")
        DV.plot_accuracy(df["iteration"], df["train_acc"], df["val_acc"], plot_name="model_accuracy.png")

if __name__ == "__main__":
    # Prepare audio data for analysis
    DataPreparation(plot_mode=False, play_mode=False, print_mode=False, test_mode=True)

    # Prepare final data for modelling
    DataFinal(plot_mode=False, print_mode=False)

    # Split datasets for modelling
    train_df, val_df, test_df = DataSplit(print_mode=False)

    # Model training and prediction
    XM = XGBoostModel(train_df, val_df, test_df)
    Modelling(plot_mode=False, print_mode=True, hyperparam_mode=False)