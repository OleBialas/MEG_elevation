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
        seq = sequence(np.asarray(cfg["speakers"], dtype=int),cfg["perception_test_trials"])
        np.savetxt(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"]+"/lists/"+condition+".txt")
    for block in cfg["meg_blocks"]:
        seq = oneback(speakers,cfg["meg_test_trials"], cfg["oneback_frequency"])
        np.savetxt(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"]+"/lists/meg_test_block_"+str(block)+".txt", seq)


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

def oneback(trial_list, repetitions, oneback_freq, space=1):

    n_trials = repetitions*len(trial_list)
    n_targets = int(n_trials * oneback_freq) # number of targets
    n_targets_per_condition = int(n_targets / len(trial_list))
    n_nontargets = n_trials - n_targets # number of nontargets
    print("oneback task contains "+str(n_targets)+" targets")
    # check if the number of targets is a multiple of the number of different conditions
    if not n_targets % len(trial_list) == 0:
        print("number of targets must be a multiple of the trial list")
        return
    # generate normal stimulus sequence:
    seq = sequence(trial_list, repetitions=int(n_nontargets / len(trial_list)), space=space)
    all_targets = np.empty(0)
    for i in np.sort(trial_list):
        indices = np.where(np.array(seq)==i)
        ok=0
        while ok == 0:
            target_indices = np.sort(np.random.choice(indices[0], n_targets_per_condition,replace=False))
            if min(np.abs(np.diff(np.sort(np.concatenate([all_targets,target_indices])))))>=space:
                ok = 1
                seq = np.insert(seq, target_indices, i)
                all_targets = np.where(np.diff(seq)==0)
                all_targets = all_targets[0]+1
            else:
                print("shuffeling")
    return seq

if __name__ == "__main__":
    seq = oneback([21,23,25,27],120,0.1)
    print(seq)
    print(len(seq))

    