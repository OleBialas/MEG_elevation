# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:47:23 2017
@author: ob56dape

Creates a list of randomized repetitions of a given list of integers
and spaces the elements as specified
"""

import numpy as np
import random
import json
import os


def prepare_experiment():
    """
    Make subject folders and generate stimulus sequences
    """
    dir = os.environ["EXPDIR"]+os.environ["SUBJECT"]

def generate_sequences():

    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    speakers=[]
    for i in cfg["speakers"]:
        speakers.append(int(i))
    for condition in cfg["perception_test_conditions"]:
        seq = sequence(speakers,cfg["perception_test_trials"])
        np.savetxt(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"]+"/lists/"+condition+".txt", seq)
        print("save...")

    return
def sequence(trial_list, repetitions, space = 1):

    seq = []  # Dictionary für erstellte Sequenz
    # check if the file already exists, if so either cancel the script or overwrite old file

    for r in range(repetitions):

        ok = 0
        print(r)
        while ok == 0:
            random.shuffle(trial_list)
            print("shuffeling...")
            
            if r > 0:
                # check if subsequent sets are sapced properly
                if np.absolute(trial_list[0] - seq[-1]) >= space and min(np.absolute(np.diff(trial_list))) >= space:
                    ok = 1
            else:
                # absolute numerical difference between subsequent elements must be greater than space
                if min(np.absolute(np.diff(trial_list))) >= space: ok = 1
        seq = seq + trial_list
    return seq

if __name__ == "__main__":
    os.environ["SUBJECT"] = "el80"  # <-- Enter Subject here
    os.environ["EXPDIR"] = "C:/Projects/MEG_Elevation/"
    generate_sequences()


    