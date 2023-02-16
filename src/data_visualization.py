import os

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import playsound
import seaborn as sns


class DataVisualization:
    """
    The DataVisualization class is used to visualize data
    """

    def __init__(self, plot_path: str):
        """
        Initialize the DataVisualization class

        :param plot_path: str, directory where the plots will be saved
        """
        self.plot_path = plot_path

    def plot_audio(self, sr: int, audio_data: np.ndarray, plot_name: str) -> None:
        """
        Save the audio signal plot with seconds on the x-axis and save it to the specified plot path

        :param sr: int, sample rate for the audio recording
        :param audio_data: np.ndarray, audio data to be plotted
        :param plot_name: str, name of the plot to be saved
        """
        # plot the audio signal
        librosa.display.waveplot(audio_data, sr=sr)

        # add labels
        plt.title("Audio Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # Clear the current figure
        plt.clf()

    def plot_stft(self, sr: int, audio_data: np.ndarray, plot_name: str) -> None:
        """
        Plot the stft signal and save it to the specified plot path

        :param sr: int, sample rate for the audio recording
        :param audio_data: np.ndarray, audio data to be plotted
        :param plot_name: str, name of the plot to be saved
        """
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
        Plots a correlation matrix and saves the figure to a specified path

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
        Save the loss plot with iteration on the x-axis and loss on the y-axis and save it to the specified plot path

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
        Save the accuracy plot with iteration on the x-axis and accuracy on the y-axis and save it to the specified plot path

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

    def column_distribution(self, df: pd.DataFrame, plot_name: str) -> None:
        """
        Plot the column distribution of the dataframe and save it to the specified plot path

        :param df: pd.DataFrame, dataframe to be plotted
        :param plot_name: str, name of the plot to be saved
        """
        # create subplots for each column of the dataframe
        df.plot.hist(subplots=True, layout=(-1, 3), sharex=False, figsize=(10, 10))

        # add a title to the plot
        plt.suptitle("Column Distribution")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()

    def plot_feature_importance(
        self, feature_importance: list, columns: list, plot_name: str
    ) -> None:
        """
        Plot the feature importance of the dataset and save it to the specified plot path

        :param feature_importance: list[float], feature importance values
        :param columns: list[str], column names to be used as x-axis labels
        :param plot_name: str, name of the plot to be saved
        """
        # create the bar chart
        plt.bar(columns, feature_importance)

        # add a title to the plot
        plt.title("Feature Importance")

        # save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # clear the current figure
        plt.clf()
