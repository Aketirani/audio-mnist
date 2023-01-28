import librosa
import numpy as np
import statistics
import scipy.stats

class DataProcessing:
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
        return librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr = self.target_sr, res_type="scipy")
    
    def zero_pad(self, data: np.ndarray) -> np.ndarray:
        """
        Pad the given audio data with zeros to the specified length
        
        :param data: np.ndarray, audio data
        :return: np.ndarray, the zero-padded audio data
        """
        padding_length = self.target_sr
        if len(data) < padding_length:
            embedded_data = np.zeros(padding_length)
            offset = np.random.randint(low = 0, high = padding_length - len(data))
            embedded_data[offset:offset+len(data)] = data
        elif len(data) == padding_length:
            embedded_data = data
        elif len(data) > padding_length:
            raise ValueError(f"Data length {len(data)} cannot exceed padding length {padding_length}!")
        return embedded_data
    
    @staticmethod
    def fft_data(data: np.ndarray) -> np.ndarray:
        """
        Calculate the FFT of the given audio data
        
        :param data: np.ndarray, audio data
        :return: np.ndarray, the FFT of the audio data
        """
        return np.fft.fft(data)

    @staticmethod
    def feature_creation(fft_data: np.ndarray) -> dict:
        """
        Calculate various statistical features of the given FFT data
        
        :param fft_data: np.ndarray, FFT data
        :return: dict, containing created features
        """
        # Take the absolute value of the FFT data
        fft_data = np.abs(fft_data)
        features = {}
        # Mean of the FFT data
        features["mean"] = np.mean(fft_data)
        # Standard deviation of the FFT data
        features["std"] = np.std(fft_data)
        # Median of the FFT data
        features["median"] = statistics.median(fft_data)
        # Maximum value of the FFT data
        features["max"] = max(fft_data)
        # Minimum value of the FFT data
        features["min"] = min(fft_data)
        # Skewness of the FFT data
        features["skewness"] = scipy.stats.skew(fft_data)
        # Kurtosis of the FFT data
        features["kurtosis"] = scipy.stats.kurtosis(fft_data)
        # Range of the FFT data
        features["dfrange"] = features["max"] - features["min"]
        # Modulation index of the FFT data
        features["modindx"] = np.std(np.diff(fft_data))/np.mean(np.diff(fft_data))
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
            features[key] = (features[key] - min(features.values()))/(max(features.values())-min(features.values()))
        return features

    @staticmethod
    def add_gender(features: dict, gender: str) -> dict:
        """
        Add the gender to the feature dict
        
        :param features: dict, containing the statistical features of the FFT data
        :param gender: str, the gender to add to the dict
        :return: dict, containing the statistical features of the FFT data with the added "gender" key
        """
        features["gender"] = gender
        return features

    @staticmethod
    def add_digit(features: dict, digit: str) -> dict:
        """
        Add the digit to the feature dict
        
        :param features: dict, containing the statistical features of the FFT data
        :param digit: str, the digit to add to the dict
        :return: dict, containing the statistical features of the FFT data with the added "digit" key
        """
        features["digit"] = digit
        return features
