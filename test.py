from __future__ import division
import numpy as np
from scipy.io import wavfile
from LBG import EUDistance
from train import training
import matplotlib.pyplot as plt
import os
from process_mfcc import mfcc_feature


nSpeaker = 40
nfiltbank = 13
(codebooks_mfcc) = training(nfiltbank,nSpeaker)
directory = os.getcwd() + '/test';
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

def predict(IdSpeaker):
    frame = str()
    fname = '/s' + str(IdSpeaker) + '.wav'
    # print('Now speaker ', str(speakerN), 'features are being tested')
    (fs, s) = wavfile.read(directory + fname)
    mel_coefs = mfcc_feature(fs,s)
    sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
    print('Speaker ', IdSpeaker, ' in test matches with speaker ', (sp_mfcc + 1), ' in train for training with MFCC')
    plt.matshow(mel_coefs)
    plt.title('MFCC input')
    plt.show()
def compare_rating():
    nCorrect_MFCC = 0
    for i in range(nSpeaker):
        fname = '/s' + str(i+1) + '.wav'
        (fs,s) = wavfile.read(directory + fname)
        mel_coefs = mfcc_feature(fs,s)
        sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
        print('Speaker ', (i+1), ' in test matches with speaker ', (sp_mfcc+1), ' in train for training with MFCC')
        if i == sp_mfcc:
            nCorrect_MFCC += 1
    percentageCorrect_MFCC = (nCorrect_MFCC/nSpeaker)*100
    # print(nCorrect_MFCC)
    print('Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%')
"""bắt đầu nhận dạng"""
# ip = input("Enter speaker name: ")
# predict(ip)
compare_rating()