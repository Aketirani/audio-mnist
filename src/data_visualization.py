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
import ydata_profiling
from sklearn.metrics import confusion_matrix


class DataVisualization:
    """
    This class is used to visualize data
    """

    def __init__(self, plot_path: str, html_path: str):
        """
        Initialize the class

        :param plot_path: str, directory where the plots will be saved
        :param html_path: str, directory where the html will be saved
        """
        self.plot_path = plot_path
        self.html_path = html_path

    def plot_audio(
        self, sr: int, audio_data: np.ndarray, plot_name: str, plot_flag: int = 1
    ) -> None:
        """
        Plots the audio signal and saves it

        :param sr: int, sample rate for the audio recording
        :param audio_data: np.ndarray, audio data to be plotted
        :param plot_name: str, name of the plot to be saved
        :param plot_flag: int, flag to determine whether to plot (1) or not (0)
        """
        if plot_flag:
            audio_data = audio_data.astype(float)
            librosa.display.waveshow(audio_data, sr=sr)
            plt.title("Audio Signal")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
            plt.clf()

    def plot_stft(
        self, sr: int, audio_data: np.ndarray, plot_name: str, plot_flag: int = 1
    ) -> None:
        """
        Plots the stft signal and saves it

        :param sr: int, sample rate for the audio recording
        :param audio_data: np.ndarray, audio data to be plotted
        :param plot_name: str, name of the plot to be saved
        :param plot_flag: int, flag to determine whether to plot (1) or not (0)
        """
        if plot_flag:
            audio_data = audio_data.astype(float)
            stft = librosa.stft(audio_data)
            stft_magnitude_db = librosa.power_to_db(np.abs(stft), ref=np.max)
            librosa.display.specshow(
                stft_magnitude_db, sr=sr, x_axis="time", y_axis="hz", cmap="plasma"
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title("Short-Time Fourier Transform")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.ylim(0, 1000)
            plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
            plt.clf()

    def plot_corr_matrix(self, corr_matrix: pd.DataFrame, plot_name: str) -> None:
        """
        Plots the correlation matrix and saves it

        :param corr_matrix: pd.DataFrame, correlation matrix to plot
        :param plot_name: str, name of the plot to be saved
        """
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            annot_kws={"size": 12},
            cbar_kws={"shrink": 0.22},
        )
        plt.title("Correlation Matrix")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
        plt.clf()

    def plot_loss(
        self, x: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, plot_name: str
    ) -> None:
        """
        Plots the loss and saves it

        :param x: np.ndarray, iteration data to be plotted on x-axis
        :param y_train: np.ndarray, training loss data to be plotted on y-axis
        :param y_val: np.ndarray, validation loss data to be plotted on y-axis
        :param plot_name: str, name of the plot to be saved
        """
        plt.plot(x, y_train, label="Training")
        plt.plot(x, y_val, label="Validation")
        plt.title("Log Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
        plt.clf()

    def plot_accuracy(
        self, x: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, plot_name: str
    ) -> None:
        """
        Plots the accuracy and saves it

        :param x: np.ndarray, iteration data to be plotted on x-axis
        :param y_train: np.ndarray, training accuracy data to be plotted on y-axis
        :param y_val: np.ndarray, validation accuracy data to be plotted on y-axis
        :param plot_name: str, name of the plot to be saved
        """
        plt.plot(x, y_train, label="Training")
        plt.plot(x, y_val, label="Validation")
        plt.title("Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
        plt.clf()

    @staticmethod
    def play_audio(filepath: str, play_flag: int = 1) -> None:
        """
        Plays the audio data

        :param filepath: str, path to the audio file
        :param play_flag: int, flag to determine whether to play (1) or not (0)
        """
        if play_flag:
            playsound.playsound(filepath)

    def plot_column_dist(
        self, df: pd.DataFrame, plot_name: str, target_column: str
    ) -> None:
        """
        Plots the column distribution with respect to the target column and saves it

        :param df: pd.DataFrame, dataframe to be plotted
        :param plot_name: str, name of the plot to be saved
        :param target_column: str, the target column for distribution comparison
        """
        numeric_columns = df.select_dtypes(include=np.number).columns
        num_columns = len(numeric_columns)
        num_rows = (num_columns + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
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
        for i in range(num_columns, num_rows * 2):
            fig.delaxes(axes[i])
        plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
        plt.clf()

    def plot_column_box(
        self, df: pd.DataFrame, plot_name: str, target_column: str
    ) -> None:
        """
        Plots the box plot with respect to the target column and saves it

        :param df: pd.DataFrame, dataframe to be plotted
        :param plot_name: str, name of the plot to be saved
        :param target_column: str, the target column for distribution comparison
        """
        numeric_columns = df.select_dtypes(include=np.number).columns
        num_columns = len(numeric_columns)
        num_rows = (num_columns + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        for i, col in enumerate(numeric_columns):
            ax = axes[i]
            sns.boxplot(
                data=df,
                x=target_column,
                y=col,
                palette="pastel",
                ax=ax,
            )
            ax.set_title(f"Box Plot of {col}")
            ax.set_xlabel(target_column)
            ax.set_ylabel(col)
        for i in range(num_columns, num_rows * 2):
            fig.delaxes(axes[i])
        plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
        plt.clf()

    def plot_feature_importance(
        self, xgb_model: xgb.XGBClassifier, plot_name: str
    ) -> None:
        """
        Plots the feature importance of the dataset and saves it

        :param xgb_model: xgb.XGBClassifier, trained XGBoost model
        :param plot_name: str, name of the plot to be saved
        """
        xgb.plot_importance(xgb_model, importance_type="weight")
        plt.title("Feature Importance")
        plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
        plt.clf()

    def plot_confusion_matrix(
        self, y_test: list, y_pred: list, labels: list, plot_name: str
    ) -> None:
        """
        Plots the confusion matrix and saves it

        :param y_test: list, true labels
        :param y_pred: list, predicted labels
        :param labels: list of str, labels for display
        :param plot_name: str, name of the plot to be saved
        """
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
        plt.clf()

    def plot_shapley_summary(
        self, xgb_model, test_X: pd.DataFrame, plot_name: str
    ) -> None:
        """
        Plots the Shapley summary and saves it

        :param xgb_model: trained XGBoost model
        :param test_X: pd.DataFrame, test features
        :param plot_name: str, name of the plot to be saved
        """
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(test_X)
        shap.summary_plot(
            shap_values,
            test_X,
            feature_names=test_X.columns,
            class_names=xgb_model.classes_,
            show=False,
        )
        plt.title("Shapley Summary Plot")
        plt.savefig(os.path.join(self.plot_path, plot_name), bbox_inches="tight")
        plt.clf()

    def profiling_report(self, df: pd.DataFrame, html_name: str) -> None:
        """
        Generates a YData Profiling report and saves it

        :param df: pd.DataFrame, the DataFrame to be profiled
        :param html_name: str, the name of the HTML file to be saved
        """
        profile = ydata_profiling.ProfileReport(df, title="Profiling Report")
        profile.to_file(os.path.join(self.html_path, html_name))
