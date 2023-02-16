import statistics

import librosa
import numpy as np
import scipy.stats


class DataProcessing:
    """
    The DataProcessing class is used to prepare audio data for analysis
    """

    def __init__(self, target_sr: int) -> None:
        """
        Initialize the class with the destination path, and target sample rate

        :param target_sr: int, target sample rate for the resampled audio recordings
        """
        self.target_sr = target_sr

    def resample_data(self, fs: int, data: np.ndarray) -> np.ndarray:
        """
        Resample the given audio data to the target sample rate and return the resampled data

        :param fs: int, sample rate of the audio data
        :param data: np.ndarray, audio data
        :return: np.ndarray, resampled audio data
        """
        # convert data to floating-point
        if not isinstance(data, float) or not (
            hasattr(data, "dtype") and np.issubdtype(data.dtype, np.floating)
        ):
            data = data.astype(np.float32)

        # resample the audio data
        return librosa.core.resample(
            y=data, orig_sr=fs, target_sr=self.target_sr, res_type="scipy"
        )

    def zero_pad(self, data: np.ndarray) -> np.ndarray:
        """
        Pad the given audio data with zeros to the specified length

        :param data: np.ndarray, audio data
        :return: np.ndarray, the zero-padded audio data
        """
        # padding length set to target sample rate for the resampled audio recordings
        padding_length = self.target_sr
        if len(data) < padding_length:
            # if the data is shorter than the target length, pad the data with zeros
            embedded_data = np.zeros(padding_length)
            offset = np.random.randint(low=0, high=padding_length - len(data))
            embedded_data[offset : offset + len(data)] = data
        elif len(data) == padding_length:
            # if the data is already of the target length, no padding is needed
            embedded_data = data
        elif len(data) > padding_length:
            # if the data is longer than the target length, raise an error
            raise ValueError(
                f"Data length {len(data)} cannot exceed padding length {padding_length}!"
            )

        # return embedded data
        return embedded_data

    @staticmethod
    def fft_data(data: np.ndarray) -> np.ndarray:
        """
        Calculate the FFT of the given audio data

        :param data: np.ndarray, audio data
        :return: np.ndarray, the FFT of the audio data
        """
        # perform FFT on the input data
        fft_data = np.fft.fft(data)

        # return FFT data
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
        # calculate the cutoff frequencies
        low_cutoff = low_threshold / self.target_sr
        high_cutoff = high_threshold / self.target_sr

        # create bandpass filter
        b, a = scipy.signal.butter(
            4, [low_cutoff, high_cutoff], btype="band", analog=False, output="ba"
        )

        # apply bandpass filter to the FFT data
        filtered_fft_data = scipy.signal.lfilter(b, a, fft_data)

        # return filtered FFT data
        return filtered_fft_data

    @staticmethod
    def feature_creation(fft_data: np.ndarray) -> dict:
        """
        Calculate various statistical features of the given FFT data

        :param fft_data: np.ndarray, FFT data
        :return: dict, containing created features
        """
        # take the absolute value of the FFT data
        fft_data = np.abs(fft_data)

        # create empty dictionary
        features = {}

        # mean
        features["mean"] = np.mean(fft_data)

        # standard deviation
        features["std"] = np.std(fft_data)

        # median
        features["med"] = statistics.median(fft_data)

        # 25th percentiles
        features["q25"] = np.percentile(fft_data, 25)

        # 75th percentiles
        features["q75"] = np.percentile(fft_data, 75)

        # minimum value
        features["min"] = min(fft_data)

        # maximum value
        features["max"] = max(fft_data)

        # skewness
        features["skew"] = scipy.stats.skew(fft_data)

        # kurtosis
        features["kurt"] = scipy.stats.kurtosis(fft_data)

        # spectral flatness
        features["sfm"] = scipy.stats.gmean(fft_data) / np.mean(fft_data)

        # frequency centroid
        features["cent"] = scipy.stats.mstats.gmean(fft_data)

        # return features
        return features

    @staticmethod
    def normalize_features(features: dict) -> dict:
        """
        Normalize the features by min-max normalization

        :param features: dict, containing the statistical features of the FFT data
        :return: dict, containing the normalized statistical features of the FFT data
        """
        # loop through all the keys in the feature dictionary
        feature_keys = list(features.keys())
        for key in feature_keys:
            # normalize the feature by min-max normalization
            features[key] = (features[key] - min(features.values())) / (
                max(features.values()) - min(features.values())
            )

        # return normalized features
        return features
