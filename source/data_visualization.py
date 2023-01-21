import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import playsound

class DataVisualization:
    def __init__(self):
        """
        Initialize the DataVisualization class
        """
        pass

    @staticmethod
    def plot_signal(sr: int, audio_data: np.ndarray, plot_path: str, plot_name: str) -> None:
        """
        Save the audio signal plot with seconds on the x-axis and save it to the specified plot path.
        
        :param sr: int, sample rate for the audio recording
        :param audio_data: np.ndarray, audio data to be plotted
        :param plot_path: str, directory where the plot will be saved
        :param plot_name: str, name of the plot to be saved
        """        
        # Plot the audio signal
        librosa.display.waveplot(audio_data, sr=sr)
        
        # Add labels to the x and y axis
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(plot_path, plot_name))
        plt.clf()

    @staticmethod
    def play_audio(filepath: str) -> None:
        """
        Play the audio data
        
        :param filepath: str, path to the audio file
        """
        playsound.playsound(filepath)

    @staticmethod
    def column_distribution(df: pd.DataFrame, plot_path: str, plot_name: str) -> None:
        """
        Plot the column distribution of the dataframe and save it to the specified plot path.
        
        :param df: pd.DataFrame, dataframe to be plotted
        :param plot_path: str, directory where the plot will be saved
        :param plot_name: str, name of the plot to be saved
        """
        df.plot.hist(subplots=True, layout=(-1, 3), sharex=False, figsize=(10,10))
        plt.savefig(os.path.join(plot_path, plot_name))
        plt.clf()
