import librosa
import numpy as np
import statistics
import scipy.stats

class DataProcessing:
    def __init__(self, dst: str, target_sr: int) -> None:
        """
        Initialize the class with the destination path, and target sample rate
        
        :param dst: str, path to the destination folder to save the resampled audio recordings
        :param target_sr: int, target sample rate for the resampled audio recordings
        """
        self.dst = dst
        self.target_sr = target_sr

    def resample_data(self, fs: int, data: np.ndarray, target_sr: int) -> np.ndarray:
        """
        Resample the given audio data to the target sample rate and return the resampled data
        
        :param fs: int, sample rate of the audio data
        :param data: np.ndarray, audio data
        :param target_sr: int, target sample rate for the resampled data
        :return: np.ndarray, resampled audio data
        """
        return librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr = target_sr, res_type="scipy")
    
    def zero_pad(self, data: np.ndarray, padding_length: int) -> np.ndarray:
        """
        Pad the given audio data with zeros to the specified length
        
        :param data: np.ndarray, audio data
        :param padding_length: int, the length to pad the audio data to
        :return: np.ndarray, the zero-padded audio data
        """
        if len(data) < padding_length:
            embedded_data = np.zeros(padding_length)
            offset = np.random.randint(low = 0, high = padding_length - len(data))
            embedded_data[offset:offset+len(data)] = data
        elif len(data) == padding_length:
            embedded_data = data
        elif len(data) > padding_length:
            raise ValueError(f"Data length {len(data)} cannot exceed padding length {padding_length}!")
        return embedded_data
    
    def fft_data(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the FFT of the given audio data
        
        :param data: np.ndarray, audio data
        :return: np.ndarray, the FFT of the audio data
        """
        return np.fft.fft(data)

    def feature_creation(self, fft_data: np.ndarray) -> dict:
        """
        Calculate various statistical features of the given FFT data
        
        :param fft_data: np.ndarray, FFT data
        :return: dict, containing the mean, std, avg, median, max, min, skewness, kurtois
        """
        fft_data = np.abs(fft_data)
        features = {}
        features["mean"] = np.mean(fft_data)
        features["std"] = np.std(fft_data)
        features["avg"] = statistics.mean(fft_data)
        features["median"] = statistics.median(fft_data)
        features["max"] = max(fft_data)
        features["min"] = min(fft_data)
        features["skewness"] = scipy.stats.skew(fft_data)
        features["kurtosis"] = scipy.stats.kurtosis(fft_data)
        return features

    def normalize_features(self, features: dict) -> dict:
        """
        Normalize the features by min-max normalization
        
        :param features: dict, containing the statistical features of the FFT data
        :return: dict, containing the normalized statistical features of the FFT data
        """
        feature_keys = list(features.keys())
        for key in feature_keys:
            features[key] = (features[key] - min(features.values()))/(max(features.values())-min(features.values()))
        return features

    def add_gender(self, features: dict, gender: str) -> dict:
        """
        Add the gender to the feature dict
        
        :param features: dict, containing the statistical features of the FFT data
        :param gender: str, the gender to add to the dict
        :return: dict, containing the statistical features of the FFT data with the added "gender" key
        """
        features["gender"] = gender
        return features
    
    def add_digit(self, features: dict, digit: str) -> dict:
        """
        Add the digit to the feature dict
        
        :param features: dict, containing the statistical features of the FFT data
        :param digit: str, the digit to add to the dict
        :return: dict, containing the statistical features of the FFT data with the added "digit" key
        """
        features["digit"] = digit
        return features
