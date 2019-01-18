# -*- coding: utf-8 -*-
"""
Evaluates response list of shape [3,n] for n trials
row1: stimulus position
row2: response
row3: reaction time

"""
import numpy as np
from matplotlib import pyplot as plt
import os
import json
from scipy.io import wavfile
from utilities import spectrum

def plot_response(response):

    #response[0] +=1
    targets = np.unique(response[:,0])
    average_error = sum(np.abs(response[:,0]-response[:,1]))/len(response)
    print("average error = "+str(average_error))
    directional_error = np.zeros(len(targets))
    count=0

    for target in targets:
        indices = np.where(response[:,0]==target)[0]
        for i in indices:
            directional_error[count] += np.abs(response[:,0][i]-response[:,1][i])
        directional_error[count] /= len(indices)
        count+=1
    print(directional_error)

    #barplot:
    ind = np.arange(len(targets))
    plt.bar(ind, directional_error, width=0.35)
    plt.show()

def plot_recordings():
    from matplotlib import pyplot as plt

    cfg = json.load(open(os.environ["EXPDIR"] +"cfg/elevation.cfg"))
    for speaker in cfg["speakers"]:
        left = wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+speaker+"_left.wav")
        right = wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+speaker+"_right.wav")
        times = np.arange(0,10,10/len(left[1]))
        Z_left, freqs, _ = spectrum(left[1], left[0],log_power=True)
        Z_right, _, _ = spectrum(right[1], right[0], log_power=True)
        fig, ax = plt.subplots(2,2, sharex="row", sharey="row")
        fig.suptitle("Speaker Number "+speaker)
        ax[0,0].plot(times, left[1])
        ax[0,1].plot(times, right[1])
        ax[1,0].plot(freqs, Z_left)
        ax[1,1].plot(freqs, Z_right)
    left = wavfile.read(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/adapter_left.wav")
    right = wavfile.read(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/adapter_right.wav")
    times = np.arange(0, 10, 10 / len(left[1]))
    Z_left, freqs, _ = spectrum(left[1], left[0], log_power=True)
    Z_right, _, _ = spectrum(right[1], right[0], log_power=True)
    fig, ax = plt.subplots(2, 2, sharex="row", sharey="row")
    fig.suptitle("Adapter")
    ax[0, 0].plot(times, left[1])
    ax[0, 1].plot(times, right[1])
    ax[1, 0].plot(freqs, Z_left)
    ax[1, 1].plot(freqs, Z_right)
