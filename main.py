#!/usr/bin/env python2

from __future__ import division
import subprocess
from reco import RecoBlock

recognizer = RecoBlock("train_data")

def menu():
    global recognizer

    # print menu 
    print "\n"
    menu_str = """Choose an option: 
                      1) Train the recognizer again
                      2) Test the trained recognizer
                      3) Exit
               """
    print menu_str
    
    # get the user's choice now
    choice = int(raw_input())

    if choice == 1:
        # train a new recognizer
        recognizer = Recoblock("train_data")
    elif choice == 2:
        # record a voice now and find out the speaker 
        subprocess.call(["arecord", "-d", "5s", "-r", "48000", "_melcache/__test.wav"]) 
        spkr = recognizer.predict("_melcache/__test.wav") 
        print "The speker must be: %s" % spkr 
    elif choice == 3:
        exit(0)
    else:
        print "Invalid choicec.\n\n"
        menu()

if __name__ == "__main__":
    menu()
