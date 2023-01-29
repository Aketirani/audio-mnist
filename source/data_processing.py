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
        # use librosa library's resample function to resample the audio data
        # the input data should be in float32 format
        # the original sample rate (fs) is passed as orig_sr
        # the target sample rate is passed as target_sr
        # res_type is set to "scipy" for high quality resampling
        return librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr = self.target_sr, res_type="scipy")
    
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
            offset = np.random.randint(low = 0, high = padding_length - len(data))
            embedded_data[offset:offset+len(data)] = data
        elif len(data) == padding_length:
            # if the data is already of the target length, no padding is needed
            embedded_data = data
        elif len(data) > padding_length:
            # if the data is longer than the target length, raise an error
            raise ValueError(f"Data length {len(data)} cannot exceed padding length {padding_length}!")
        
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

    def bandpass_filter(self, fft_data: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
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
        b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False, output='ba')
        
        # apply bandpass filter to the FFT data
        filtered_fft_data = scipy.signal.lfilter(b, a, fft_data)
        
        # return filtered FFT data
        return filtered_fft_data

    @staticmethod
    def feature_creation(audio_data: np.ndarray, fft_data: np.ndarray) -> dict:
        """
        Calculate various statistical features of the given FFT data
        
        :param audio_data: np.ndarray, audio data
        :param fft_data: np.ndarray, FFT data
        :return: dict, containing created features
        """
        # Take the absolute value of the FFT data
        fft_data = np.abs(fft_data)
        features = {}

        # mean of the data
        features["mean_audio"] = np.mean(audio_data)
        features["mean_fft"] = np.mean(fft_data)

        # standard deviation of the data
        features["std_audio"] = np.std(audio_data)
        features["std_fft"] = np.std(fft_data)

        # median of the data
        features["median_audio"] = statistics.median(audio_data)
        features["median_fft"] = statistics.median(fft_data)

        # 25th percentiles of the data
        features["Q25_audio"] = np.percentile(audio_data, 25)
        features["Q25_fft"] = np.percentile(fft_data, 25)

        # 75th percentiles of the data
        features["Q75_audio"] = np.percentile(audio_data, 75)
        features["Q75_fft"] = np.percentile(fft_data, 75)

        # minimum value of the data
        features["min_audio"] = min(audio_data)
        features["min_fft"] = min(fft_data)

        # maximum value of the data
        features["max_audio"] = max(audio_data)
        features["max_fft"] = max(fft_data)

        # skewness of the data
        features["skewness_audio"] = scipy.stats.skew(audio_data)
        features["skewness_fft"] = scipy.stats.skew(fft_data)

        # kurtosis of the data
        features["kurtosis_audio"] = scipy.stats.kurtosis(audio_data)
        features["kurtosis_fft"] = scipy.stats.kurtosis(fft_data)

        # range of the data
        features["dfrange_audio"] = np.abs(max(audio_data)-min(audio_data))
        features["dfrange_fft"] = np.abs(max(fft_data)-min(fft_data))

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
            features[key] = (features[key] - min(features.values()))/(max(features.values())-min(features.values()))
        
        # return normalized features
        return features

    @staticmethod
    def add_gender(features: dict, gender: str) -> dict:
        """
        Add the gender to the feature dict
        
        :param features: dict, containing the statistical features of the FFT data
        :param gender: str, the gender to add to the dict
        :return: dict, containing the statistical features of the FFT data with the added "gender" key
        """
        # add the gender to the feature dict with key "gender"
        features["gender"] = gender

        # return the updated feature dictionary
        return features

    @staticmethod
    def add_digit(features: dict, digit: str) -> dict:
        """
        Add the digit to the feature dict
        
        :param features: dict, containing the statistical features of the FFT data
        :param digit: str, the digit to add to the dict
        :return: dict, containing the statistical features of the FFT data with the added "digit" key
        """
        # add the digit as a key to the feature dictionary
        features["digit"] = digit
        
        # return the updated feature dictionary
        return features
