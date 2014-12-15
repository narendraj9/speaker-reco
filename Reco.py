#!/usr/bin/env python2

import csv
import os.path
import numpy as np
import scipy.io.wavfile as wavfile

from sklearn import svm
from scikits.talkbox.features import mfcc

class RecoBlock():
    """Recognizer [If there is a word like that] Block. This class
    keeps the state required to recognize a set of speakers after it has
    been trained.
    """

    def __init__(self, data_dir, out_dir="_melcache"):
        self.data_dir = os.path.abspath(data_dir);
        self.recognizer = svm.LinearSVC()
        
        # make folder for caching
        os.mkdir(os.path.join(os.getcwd(), "_melcache"))
        train_file = os.path.join(os.getcwd(), "_melcache", "training_data.csv")

        # generate training dataset csv file and get data for training
        self._gen_mfccs(self.data_dir, train_file)
        melv_list, speaker_ids = self._get_tdata(train_file)

        # train a linear svm now
        self.recognizer.fit(melv_list, speaker_ids)
        
        
    def _gen_mfccs(self, data_dir, outfile):
        """ Generates a csv file containing labeled lines for each speaker """

        with open(outfile, 'w') as ohandle:
            melwriter = csv.writer(ohandle)
            speakers = os.listdir(data_dir)
         
        for skr_dir in speakers:
            for soundclip in os.listdir(os.path.join(data_dir, skr_dir)):
                # generate mel coefficients for the current clip
                clip_path = os.path.abspath(os.path.join(data_dir, skr_dir, soundclip))
                sample_rate, data = wavfile.read(clip_path)
                ceps, mspec = mfcc(data)
                
                # write an entry into the training file for the current speaker
                melwrite.writerow(ceps + [skr_dir])

    def predict(self, soundclip):
        """ Recognizes the speaker in the sound clip. """

        sample_rate, data = wavfile.read(os.path.abspath(soundclip))
        melvector = mfcc(data)

        return self.recognizer.predict(melvector)
        
    
    def _get_tdata(self, icsv):
        """ Returns the input and output example lists to be sent to an SVM
        classifier.
        """
        melv_list = []
        speaker_ids = []
        with open(icsv, 'r') as icsv_handle:
            melreader = csv.reader(icsv_handle)
            
            for example in melreader:
                melv_list.append(map(float, example[:-1]))
                speaker_ids.append(example[-1])
                
        return melv_list, speaker_ids
        
if __name__ == "__main__":
    recoblock = RecoBlock("train_data")
    print recoblock.predict("./test_data/test1.wav")
    
