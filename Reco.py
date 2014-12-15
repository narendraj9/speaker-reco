#!/usr/bin/env python2

from __future__ import division

import csv
import os.path
import logging
import numpy as np
import scipy.io.wavfile as wavfile

from sklearn import svm
from scikits.talkbox.features import mfcc
from sklearn.externals import joblib

logging.basicConfig(level=logging.DEBUG)

class RecoBlock():
    """Recognizer [If there is a word like that] Block. This class
    keeps the state required to recognize a set of speakers after it has
    been trained.
    """

    def __init__(self, data_dir, out_dir="_melcache"):
        self.data_dir = os.path.abspath(data_dir);
        
        # make folder for storing the model
        melcache_dir = os.path.join(os.getcwd(), "_melcache")
        try:
            os.mkdir(melcache_dir)
            train_file = os.path.join(melcache_dir, "training_data.csv")
        except OSError:
            # if the folder exists try loading the saved model
            self.recognizer = joblib.load(os.path.join(melcache_dir, "recognizer.pkl"))
            return
        
        self.recognizer = svm.LinearSVC()

        # generate training dataset csv file and get data for training
        self._gen_features(self.data_dir, train_file)
        melv_list, speaker_ids = self._get_tdata(train_file)

        logging.debug(speaker_ids)

        # train a linear svm now
        self.recognizer.fit(melv_list, speaker_ids)

        # save the trained svm in a file for later use
        joblib.dump(self.recognizer, os.path.join(melcache_dir, "recognizer.pkl"))
       
    def _mfcc_to_fvec(self, ceps):
        # calculate the mean 
        mean = np.mean(ceps, axis=0)
        # and standard deviation of MFCC vectors 
        std = np.std(ceps, axis=0)
        # use [mean, std] as the feature vector
        fvec = np.concatenate((mean, std)).tolist()
        return fvec
        
    def _gen_features(self, data_dir, outfile):
        """ Generates a csv file containing labeled lines for each speaker """

        with open(outfile, 'w') as ohandle:
            melwriter = csv.writer(ohandle)
            speakers = os.listdir(data_dir)
         
            for spkr_dir in speakers:
                for soundclip in os.listdir(os.path.join(data_dir, spkr_dir)):
                    # generate mel coefficients for the current clip
                    clip_path = os.path.abspath(os.path.join(data_dir, spkr_dir, soundclip))
                    sample_rate, data = wavfile.read(clip_path)
                    ceps, mspec, spec = mfcc(data, fs=sample_rate)
                
                    # write an entry into the training file for the current speaker
                    # the vector to store in csv file contains the speaker's name at the end 
                    fvec = self._mfcc_to_fvec(ceps)
                    fvec.append(spkr_dir)

                    logging.debug(fvec) # see the numbers [as if they make sense ]

                    # write one row to the csv file
                    melwriter.writerow(fvec)

    def _get_tdata(self, icsv):
        """ Returns the input and output example lists to be sent to an SVM
        classifier.
        """
        melv_list = []
        speaker_ids = []

        # build melv_list and speaker_ids lists
        with open(icsv, 'r') as icsv_handle:
            melreader = csv.reader(icsv_handle)
            
            for example in melreader:
                melv_list.append(map(float, example[:-1]))
                speaker_ids.append(example[-1])

        # and return them!        
        return melv_list, speaker_ids
            
    def predict(self, soundclip):
        """ Recognizes the speaker in the sound clip. """

        sample_rate, data = wavfile.read(os.path.abspath(soundclip))
        ceps, mspec, spec = mfcc(data, fs=sample_rate)
        fvec = self._mfcc_to_fvec(ceps)
        
        return self.recognizer.predict(fvec)
        
if __name__ == "__main__":
    recoblock = RecoBlock("train_data")
    
    test_dir = os.path.abspath("test_data")
    testset_size = 0
    testset_error = 0

    for spkr_dir in os.listdir(test_dir):
        for soundclip in os.listdir(os.path.join(test_dir, spkr_dir)):
            clippath = os.path.abspath(os.path.join(test_dir, spkr_dir))
            prediction = recoblock.predict(clippath)

            testset_size += 1
            if prediction != spkr_dir:
                testset_error += 1    
    
    if testset_size == 0:
        print "No test data available."
    else:
        print "Error on test data: %f\n" (testset_error / testset_size * 100)
