import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import playsound

class DataVisualization:
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
        # Plot the audio signal
        librosa.display.waveplot(audio_data, sr=sr)
        
        # Add labels to the x and y axis
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))

        # Clear the current figure
        plt.clf()

    def plot_stft(self, sr: int, audio_data: np.ndarray, plot_name: str) -> None:
        """
        Plot the STFT signal and save it to the specified plot path
        
        :param sr: int, sample rate for the audio recording
        :param audio_data: np.ndarray, audio data to be plotted
        :param plot_name: str, name of the plot to be saved
        """
        # Compute the STFT of the audio signal
        stft = librosa.stft(audio_data)
        
        # Get the magnitude of the STFT data
        stft_magnitude = np.abs(stft)
        
        # Convert the magnitude to dB scale
        stft_magnitude_db = librosa.power_to_db(stft_magnitude, ref=np.max)
        
        # Use librosa to display the amplitude of the stft data in dB on the y-axis and time on the x-axis
        librosa.display.specshow(stft_magnitude_db, sr=sr, x_axis='time', y_axis='hz', cmap='inferno')

        # Add a colorbar with dB scale
        plt.colorbar(format='%+2.0f dB')
        plt.title('STFT spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim(0, 1000)
        
        # Save the plot to the specified filepath with the given file name
        plt.savefig(os.path.join(self.plot_path, plot_name))
        
        # Clear the current figure
        plt.clf()

    @staticmethod
    def play_audio(filepath: str) -> None:
        """
        Play the audio data
        
        :param filepath: str, path to the audio file
        """
        playsound.playsound(filepath)

    def column_distribution(self, df: pd.DataFrame, plot_name: str) -> None:
        """
        Plot the column distribution of the dataframe and save it to the specified plot path
        
        :param df: pd.DataFrame, dataframe to be plotted
        :param plot_name: str, name of the plot to be saved
        """
        df.plot.hist(subplots=True, layout=(-1, 3), sharex=False, figsize=(10,10))
        plt.savefig(os.path.join(self.plot_path, plot_name))
        plt.clf()
