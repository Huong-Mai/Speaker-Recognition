from __future__ import division
import numpy as np
from LBG import EUDistance
from process_voice import readfile, show_mfcc
from train import training
import os


nfiltbank = 13
directory = os.getcwd() + '/test/';
(codebooks_mfcc) = training(nfiltbank)

def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k,:,:])
        dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0]) 
        if dist < distmin:
            distmin = dist
            speaker = k
    return speaker

def predict():
    # Nhận dạng
    file_input = input("Nhập tên file âm thanh cần nhận dạng: ")
    mel_coefs = readfile(directory,file_input)
    sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
    print('Speaker ', file_input, ' in test matches with speaker ', (sp_mfcc + 1), ' in train for training with MFCC')
    # plt.matshow(mel_coefs)
    # plt.title('MFCC input')
    # plt.show()
    show_mfcc(mel_coefs)

def compare_rating():
    nSpeaker = input('Nhập số lượng giọng nói cần nhận dạng: ')
    nCorrect_MFCC = 0
    for i in range(nSpeaker):
        fname = 's'+str(i+1)
        mel_coefs = readfile(directory,fname)
        sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
        print(mel_coefs.shape)
        print('Speaker ', (i+1), ' in test matches with speaker ', (sp_mfcc+1), ' in train for training with MFCC')
        if i == sp_mfcc:
            nCorrect_MFCC += 1
    percentageCorrect_MFCC = (nCorrect_MFCC/nSpeaker)*100
    print('Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%')

