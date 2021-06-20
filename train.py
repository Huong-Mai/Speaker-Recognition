from __future__ import division
import numpy as np
from LBG import lbg
import os

from process_voice import readfile

def training(nfiltbank):
    directory = os.getcwd() + '/train/';
    nSpeaker = len(os.listdir(directory))
    nCentroid = 16
    codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
    directory = os.getcwd() + '/train/';
    fname = str()
    for i in range(nSpeaker):
        fname = 's' + str(i+1)
        mel_coeff = readfile(directory,fname)
        codebooks_mfcc[i,:,:] = lbg(mel_coeff, nCentroid)
    print('Training complete')
    return (codebooks_mfcc)
