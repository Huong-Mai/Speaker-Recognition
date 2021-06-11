import numpy as np
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt

def mfcc_feature(frequency_sampling, audio_signal):
   # frequency_sampling, audio_signal = wavfile.read(pathinput)

   audio_signal = audio_signal / np.power(2, 15)
   length_signal = len(audio_signal)
   half_length = np.ceil((length_signal + 1) / 2.0).astype(int)

   signal_frequency = np.fft.fft(audio_signal)
   signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
   signal_frequency **= 2

   len_fts = len(signal_frequency)

   if length_signal % 2:
      signal_frequency[1:len_fts] *= 2
   else:
      signal_frequency[1:len_fts-1] *= 2

   audio_signal = audio_signal[:15000]
   features_mfcc = mfcc(audio_signal, frequency_sampling)
   features_mfcc = features_mfcc.T

   return features_mfcc
def show_mfccFB(frequency_sampling, audio_signal):
   filterbank_features = logfbank(audio_signal, frequency_sampling)
   print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
   print('Length of each feature =', filterbank_features.shape[1])
def show_mfcc(frequency_sampling, audio_signal):
   features_mfcc = mfcc_feature(frequency_sampling,audio_signal)
   plt.matshow(features_mfcc)
   plt.title('MFCC')
   plt.show()