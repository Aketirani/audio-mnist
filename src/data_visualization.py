import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import playsound
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import confusion_matrix


class DataVisualization:
    """
    This class is used to visualize data
    """

    def __init__(self, plot_path: str):
        """
        Initialize the class

        :param plot_path: str, directory where the plots will be saved
        """
        self.plot_path = plot_path

    def plot_audio(self, sr: int, audio_data: np.ndarray, plot_name: str) -> None:
        """
        Plots the audio signal with seconds on the x-axis and saves it

        :param sr: int, sample rate for the audio recording
        :param audio_data: np.ndarray, audio data to be plotted
        :param plot_name: str, name of the plot to be saved
        """
        # convert audio data to floating-point format
        audio_data = audio_data.astype(float)

        # plot the audio signal
        librosa.display.waveshow(audio_data, sr=sr)

        # add labels
        plt.title("Audio Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    def plot_stft(self, sr: int, audio_data: np.ndarray, plot_name: str) -> None:
        """
        Plots the stft signal and saves it

        :param sr: int, sample rate for the audio recording
        :param audio_data: np.ndarray, audio data to be plotted
        :param plot_name: str, name of the plot to be saved
        """
        # convert audio data to floating-point format
        audio_data = audio_data.astype(float)

        # compute the stft of the audio signal
        stft = librosa.stft(audio_data)

        # get the magnitude of the stft data
        stft_magnitude = np.abs(stft)

        # convert the magnitude to dB scale
        stft_magnitude_db = librosa.power_to_db(stft_magnitude, ref=np.max)

        # display the amplitude of the stft data in dB
        librosa.display.specshow(
            stft_magnitude_db, sr=sr, x_axis="time", y_axis="hz", cmap="inferno"
        )

        # add a colorbar with dB scale and labels
        plt.colorbar(format="%+2.0f dB")
        plt.title("STFT Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(0, 1000)

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    def plot_corr_matrix(self, corr_matrix: pd.DataFrame, plot_name: str) -> None:
        """
        Plots the correlation matrix and saves it

        :param corr_matrix: pd.DataFrame, correlation matrix to plot
        :param plot_name: str, name of the plot to be saved
        """
        # use seaborn to create a heatmap of the correlation matrix
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)

        # set the title of the plot
        plt.title("Correlation Matrix")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    def plot_loss(
        self, x: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, plot_name: str
    ) -> None:
        """
        Plots iteration on the x-axis and loss on the y-axis and saves it

        :param x: np.ndarray, iteration data to be plotted on x-axis
        :param y_train: np.ndarray, training loss data to be plotted on y-axis
        :param y_val: np.ndarray, validation loss data to be plotted on y-axis
        :param plot_name: str, name of the plot to be saved
        """
        # plot y_train and y_val loss over iteration
        plt.plot(x, y_train, label="Training")
        plt.plot(x, y_val, label="Validation")

        # add labels
        plt.title("Log Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    def plot_accuracy(
        self, x: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, plot_name: str
    ) -> None:
        """
        Plots iteration on the x-axis and accuracy on the y-axis and saves it

        :param x: np.ndarray, iteration data to be plotted on x-axis
        :param y_train: np.ndarray, training accuracy data to be plotted on y-axis
        :param y_val: np.ndarray, validation accuracy data to be plotted on y-axis
        :param plot_name: str, name of the plot to be saved
        """
        # plot y_train and y_val accuracy over iteration
        plt.plot(x, y_train, label="Training")
        plt.plot(x, y_val, label="Validation")

        # add labels
        plt.title("Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper right")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    @staticmethod
    def play_audio(filepath: str) -> None:
        """
        Play the audio data

        :param filepath: str, path to the audio file
        """
        # play audio data sound
        playsound.playsound(filepath)

    def plot_column_dist(
        self, df: pd.DataFrame, plot_name: str, target_column: str
    ) -> None:
        """
        Plots the column distribution of the dataframe with respect to the target column and saves it

        :param df: pd.DataFrame, dataframe to be plotted
        :param plot_name: str, name of the plot to be saved
        :param target_column: str, the target column for distribution comparison
        """
        # select numeric columns
        numeric_columns = df.select_dtypes(include=np.number).columns

        # calculate the number of columns and rows for subplot layout
        num_columns = len(numeric_columns)
        num_rows = (num_columns + 1) // 2

        # create a grid of subplots with custom layout
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        # loop through columns and plot histograms
        for i, col in enumerate(numeric_columns):
            ax = axes[i]
            sns.histplot(
                data=df,
                x=col,
                hue=target_column,
                common_norm=False,
                palette="pastel",
                ax=ax,
            )
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel("")
            ax.set_ylabel("Count" if col == target_column else "Density")

        # hide any empty subplots
        for i in range(num_columns, num_rows * 2):
            fig.delaxes(axes[i])

        # adjust layout and spacing
        plt.tight_layout()

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    def plot_feature_importance(
        self, xgb_model: xgb.XGBClassifier, plot_name: str
    ) -> None:
        """
        Plots the feature importance of the dataset and saves it

        :param xgb_model: xgb.XGBClassifier, trained XGBoost model
        :param plot_name: str, name of the plot to be saved
        """
        # plot feature importance
        xgb.plot_importance(xgb_model, importance_type="weight")

        # add title to the plot
        plt.title("Feature Importance")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    def plot_confusion_matrix(
        self, y_test: list, y_pred: list, labels: list, plot_name: str
    ) -> None:
        """
        Plots the confusion matrix of the classification and saves it

        :param y_test: list, true labels
        :param y_pred: list, predicted labels
        :param labels: list of str, labels for display
        :param plot_name: str, name of the plot to be saved
        """
        # compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # use seaborn to create a heatmap of the confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )

        # add labels
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    def plot_shapley_summary(
        self, xgb_model, test_X: pd.DataFrame, plot_name: str
    ) -> None:
        """
        Plots the Shapley summary of the classification and saves it

        :param xgb_model: trained XGBoost model
        :param test_X: pd.DataFrame, test features
        :param plot_name: str, name of the plot to be saved
        """
        # create a TreeExplainer for the XGBoost model
        explainer = shap.TreeExplainer(xgb_model)

        # calculate Shapley values
        shap_values = explainer.shap_values(test_X)

        # create the Shapley summary plot
        shap.summary_plot(
            shap_values,
            test_X,
            feature_names=test_X.columns,
            class_names=xgb_model.classes_,
            show=False,
        )

        # add title to the plot
        plt.title("Shapley Summary Plot")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()
