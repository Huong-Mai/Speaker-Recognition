from __future__ import division
import numpy as np
from scipy.io import wavfile
from LBG import lbg
from process_mfcc import mfcc_feature
import os


def training(nfiltbank,nSpeaker):
    nCentroid = 16
    codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
    directory = os.getcwd() + '/train';
    fname = str()
    for i in range(nSpeaker):
        fname = '/s' + str(i+1) + '.wav'
        (fs, s) = wavfile.read(directory + fname)
        # print('Now speaker ', str(i+1), 'features are being trained' )
        mel_coeff = mfcc_feature(fs,s)
        codebooks_mfcc[i,:,:] = lbg(mel_coeff, nCentroid)
    print('Training complete')
    return (codebooks_mfcc)
