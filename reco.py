#!/usr/bin/env python2

from __future__ import division

import csv
import os.path
import logging
import numpy as np
import scipy.io.wavfile as wavfile

from sklearn import svm
from features import mfcc

logging.basicConfig(level=logging.DEBUG)

class RecoBlock():
    """Recognizer [If there is a word like that] Block. This class
    keeps the state required to recognize a set of speakers after it has
    been trained.
    """

    def __init__(self, data_dir, out_dir="_melcache"):
        self.data_dir = os.path.abspath(data_dir);
        
        # make folder for storing the csv file holding training data
        melcache_dir = os.path.join(os.getcwd(), "_melcache")
        train_file = os.path.join(melcache_dir, "training_data.csv")
        try:
            os.mkdir(melcache_dir)
            # generate training dataset csv file and get data for training
            self._gen_features(self.data_dir, train_file)
        except OSError:
            logging.debug("_melcache already exists. Assuming training_data.csv exists too.")

        self.recognizer = svm.LinearSVC()
        melv_list, speaker_names = self._get_tdata(train_file)
        
        # generate speaker_ids from speaker_names
        self.spkr_ntoi = {}
        self.spkr_iton = {}

        i = 0 
        for name in speaker_names:
            if name not in self.spkr_ntoi:
               self.spkr_ntoi[name] = i
               self.spkr_iton[i] = name
               i += 1
        
        speaker_ids = map(lambda n: self.spkr_ntoi[n], speaker_names)

        logging.debug(speaker_ids)

        # train a linear svm now
        self.recognizer.fit(melv_list, speaker_ids)

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
                    ceps = mfcc(data, sample_rate)
                
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
        ceps = mfcc(data, sample_rate)
        fvec = self._mfcc_to_fvec(ceps)
        
        speaker_id = self.recognizer.predict(fvec)[0]

        return self.spkr_iton[speaker_id]
        
if __name__ == "__main__":
    recoblock = RecoBlock("train_data")
    
    test_dir = os.path.abspath("test_data")
    testset_size = 0
    testset_error = 0

    for spkr_dir in os.listdir(test_dir):
        for soundclip in os.listdir(os.path.join(test_dir, spkr_dir)):
            clippath = os.path.abspath(os.path.join(test_dir, spkr_dir, soundclip))
            prediction = recoblock.predict(clippath)
            
            testset_size += 1
            if prediction != spkr_dir:
                testset_error += 1    
                print "%s %s " % (prediction, u"[\u2717]")
            else:
                print "%s %s " % (prediction, u"[\u2713]")

    if testset_size == 0:
        print "No test data available."
    else:
        print "Error on test data: %.2f%%\n" % (testset_error / testset_size * 100)
