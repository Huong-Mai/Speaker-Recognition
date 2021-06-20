import os
import wave
import pyaudio
import matplotlib.pyplot as plt
from scipy.io import wavfile
from code_mfcc_feature import mfcc

def readfile(directory,filename):
    fname = directory + filename +'.wav'
    (fs, s) = wavfile.read(fname)
    s = s[0:int(2 * fs)]  # Keep the first 2 seconds
    mel_coefs = mfcc(s, fs)
    mel_coefs = mel_coefs.T
    return mel_coefs

def write_file_mic():
    # file_name = input("Nhập tên âm thanh muốn lưu: ")
    dir = input("Vị trí lưu âm thanh: ")
    directory = os.getcwd() + '/' + dir + "/"
    if dir == 'test':
        file_name = input("Nhập tên âm thanh muốn lưu: ")
    elif dir == 'train':
        file_name = 's' + str( len(os.listdir(directory))+1 )
    microphone_integration(directory, file_name)
    return 0

def microphone_integration(directory,file_name):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = directory + file_name + ".wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def show_mfcc(features_mfcc):
   print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
   print('Length of each feature =', features_mfcc.shape[1])
   plt.matshow(features_mfcc)
   plt.title('MFCC')
   plt.show()