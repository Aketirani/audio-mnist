import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavf
import scipy.stats


class DataPreparation:
    """
    This class is used to prepare data for analysis
    """

    def __init__(self, target_sr: int = 8000) -> None:
        """
        Initialize the class with the destination path, and target sample rate

        :param target_sr: int, target sample rate for the resampled audio recordings
        """
        self.target_sr = target_sr

    @staticmethod
    def read_audio(filepath: str) -> tuple:
        """
        Read the audio data from the given file and return the sample rate and audio data

        :param filepath: str, path to the audio file
        :return audio: tuple, containing the sample rate and audio data
        """
        try:
            audio = wavf.read(filepath)
        except:
            raise FileNotFoundError(f"{filepath} is not a valid filepath!")
        return audio

    def resample_data(self, fs: int, data: np.ndarray) -> np.ndarray:
        """
        Resample the given audio data to the target sample rate and return the resampled data

        :param fs: int, sample rate of the audio data
        :param data: np.ndarray, audio data
        :return: np.ndarray, resampled audio data
        """
        if not isinstance(data, float) or not (
            hasattr(data, "dtype") and np.issubdtype(data.dtype, np.floating)
        ):
            data = data.astype(np.float32)
        resampled_data = librosa.core.resample(
            y=data, orig_sr=fs, target_sr=self.target_sr, res_type="scipy"
        )
        return resampled_data

    def zero_pad(self, data: np.ndarray) -> np.ndarray:
        """
        Pad the given audio data with zeros to the specified length

        :param data: np.ndarray, audio data
        :return: np.ndarray, the zero-padded audio data
        """
        padding_length = self.target_sr
        if len(data) < padding_length:
            embedded_data = np.zeros(padding_length)
            offset = np.random.randint(low=0, high=padding_length - len(data))
            embedded_data[offset : offset + len(data)] = data
        elif len(data) == padding_length:
            embedded_data = data
        elif len(data) > padding_length:
            raise ValueError(
                f"Data length {len(data)} cannot exceed padding length {padding_length}!"
            )
        return embedded_data

    @staticmethod
    def fft_data(data: np.ndarray) -> np.ndarray:
        """
        Calculate the FFT of the given audio data

        :param data: np.ndarray, audio data
        :return: np.ndarray, the FFT of the audio data
        """
        fft_data = np.fft.fft(data)
        return fft_data

    def bandpass_filter(
        self, fft_data: np.ndarray, low_threshold: float, high_threshold: float
    ) -> np.ndarray:
        """
        Apply bandpass filter to the given FFT data to keep frequencies between the lower and higher threshold

        :param fft_data: np.ndarray, the FFT data
        :param low_threshold: float, the lower threshold frequency
        :param high_threshold: float, the higher threshold frequency
        :return: np.ndarray, the filtered FFT data
        """
        low_cutoff = low_threshold / self.target_sr
        high_cutoff = high_threshold / self.target_sr
        b, a = scipy.signal.butter(
            4, [low_cutoff, high_cutoff], btype="band", analog=False, output="ba"
        )
        filtered_fft_data = scipy.signal.lfilter(b, a, fft_data)
        return filtered_fft_data

    @staticmethod
    def feature_creation_time_domain(audio_data: np.ndarray) -> dict:
        """
        Calculate time-domain features of the given audio data

        :param audio_data: np.ndarray, audio data
        :return: dict, containing time-domain features
        """
        features = {}
        features["mean_t"] = np.mean(audio_data)
        features["std_t"] = np.std(audio_data)
        features["med_t"] = np.median(audio_data)
        features["min_t"] = np.min(audio_data)
        features["max_t"] = np.max(audio_data)
        features["q25_t"] = np.percentile(audio_data, 25)
        features["q75_t"] = np.percentile(audio_data, 75)
        features["skew_t"] = scipy.stats.skew(audio_data)
        features["kurt_t"] = scipy.stats.kurtosis(audio_data)
        features["zeroxrate_t"] = librosa.feature.zero_crossing_rate(audio_data)[0, 0]
        features["entropy_t"] = scipy.stats.entropy(
            np.histogram(audio_data, bins=10)[0]
        )
        return features

    @staticmethod
    def feature_creation_frequency_domain(fft_data: np.ndarray) -> dict:
        """
        Calculate frequency-domain features of the given FFT data

        :param fft_data: np.ndarray, FFT data
        :return: dict, containing created features
        """
        fft_data = np.abs(fft_data)
        features = {}
        features["mean_f"] = np.mean(fft_data)
        features["std_f"] = np.std(fft_data)
        features["med_f"] = np.median(fft_data)
        features["min_f"] = min(fft_data)
        features["max_f"] = max(fft_data)
        features["q25_f"] = np.percentile(fft_data, 25)
        features["q75_f"] = np.percentile(fft_data, 75)
        features["skew_f"] = scipy.stats.skew(fft_data)
        features["kurt_f"] = scipy.stats.kurtosis(fft_data)
        features["sfm_f"] = scipy.stats.gmean(fft_data) / np.mean(fft_data)
        features["cent_f"] = scipy.stats.mstats.gmean(fft_data)
        return features

    @staticmethod
    def normalize_features(features: dict) -> dict:
        """
        Normalize the features by min-max normalization

        :param features: dict, containing the statistical features of the FFT data
        :return: dict, containing the normalized statistical features of the FFT data
        """
        feature_keys = list(features.keys())
        for key in feature_keys:
            features[key] = (features[key] - min(features.values())) / (
                max(features.values()) - min(features.values())
            )
        return features

    @staticmethod
    def add_column_dict(data: dict, column_name: str, value: str) -> dict:
        """
        Add column and value to the dictionary

        :param data: dict, input data
        :param column_name: str, column name to add
        :param value: str, value to add
        :return: dict, data with column added
        """
        data[column_name] = value
        return data

    @staticmethod
    def add_column_df(data: pd.DataFrame, column_name: str, value: str) -> pd.DataFrame:
        """
        Add column and value to the DataFrame

        :param data: DataFrame, input data
        :param column_name: str, column name to add
        :param value: str, value to add
        :return: DataFrame, data with column added
        """
        data[column_name] = value
        return data

    @staticmethod
    def remove_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove a column from DataFrame

        :param df: pd.DataFrame, input DataFrame
        :param column: str, column name to remove
        :return: pd.DataFrame, DataFrame with column removed
        """
        if column in df.columns:
            df = df.drop([column], axis=1)
        else:
            raise ValueError(f"{column} not found in DataFrame")
        return df

    @staticmethod
    def column_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
        """
        Returns the value counts of a given column in a DataFrame

        :param df: pd.DataFrame, input DataFrame
        :param column: str, name of the column in the DataFrame
        :return: pd.Series, containing the value counts of the specified column
        """
        value_counts = df[column].value_counts()
        return value_counts
