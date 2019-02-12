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
    # TODO: warum ist das spektrum vom adapter nach oben verschoben?
    cfg = json.load(open(os.environ["EXPDIR"] +"cfg/elevation.cfg"))
    for speaker in cfg["speakers"]+["adapter"]:
        fig, ax = plt.subplots(2,2, sharex="row", sharey="row")
        fig.suptitle(speaker)
        ax[0,0].set_ylim([-5,5])
        ax[1,0].set_ylim([-130,-20])
        for side, s in zip(["left", "right"], [0,1]):
            if speaker == "adapter":
                sound = wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"]+"/recordings/adapter_"+side+".wav")
            else:
                sound = wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+speaker+"_"+side+"_norm.wav")
            times = np.arange(0,10,10/len(sound[1]))
            Z, freqs, _ = spectrum(sound[1], sound[0],log_power=True)
            ax[0,s].plot(times, sound[1])
            ax[1,s].plot(freqs, Z)
    plt.show()
