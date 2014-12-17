#!/usr/bin/env python2

from __future__ import division
import
from reco import RecoBlock

def menu():
    # print menu 
    print "\n"
    menu_str = """Choose an option: 
                      1) Train a recognizer
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
        # test an existing recognizer
        recognizer = Recoblock("train_data")
        cache_dir = "_melcache"
        
    elif choice == 3:
        exit(0)
    else:
        print "Invalid choicec.\n\n"
        menu()

if __name__ == "__main__":
    menu()
