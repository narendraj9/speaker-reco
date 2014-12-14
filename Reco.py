#!/usr/bin/env python2

import os.path
import numpy as np
import scipy.io.wavefile as wavefile

from sklearn import svm
from scikits.talkbox.features import mfcc

class RecoBlock():
    """Recognizer [If there is a word like that] Block. This class
    keeps the state required to recognize a set of speakers after it has
    been trained.
    """

    def __init__(self, data_dir, out_dir="_melcache"):
        self.data_dir = os.path.abspath(data_dir);

        # make folder for caching
        os.path.mkdir(os.path.join(os.path.getcwd(), "_melcache"))
        train_file = os.path.join(os.path.getcwd(), "_melcache"))

        # generate training dataset csv file
        self._gen_mfccs(self.data_dir, train_file)


    def _gen_mfccs(self, data_dir, outfile):
        """ Generates a csv file containing labeled lines for each speaker """

        with open(outfile, 'w') as ohandle:
            melwriter = csv.writer(ohandle)
            speakers = os.listdir(data_dir)
        
        for skr_dir in speakers:
            for soundclip in os.listdir(skr_dir)
                # generate mel coefficients for the current clip
                sample_rate, data = wavefile.read(os.path.abspath(soundclip))
                ceps, mspec = mfcc(data)
                
                # write an entry into the training file for the current speaker
                melwrite.writerow(ceps + [skr_dir])
        
        # done with generating the training data file
        
        
        
        
