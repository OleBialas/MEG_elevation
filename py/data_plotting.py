from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from utilities import spectrum

def make_splitplot(xlabel='Time (ms)', ylabel='Voltage (μV)'):
    """
    make a big subplot with two small subplots inside so that there is one
    x- and y-label for both subplots. returns figure and axes
    """
    fig = plt.figure()
    ax0 = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # Turn off axis lines and ticks of the big subplot
    ax0.spines['top'].set_color('none')
    ax0.spines['bottom'].set_color('none')
    ax0.spines['left'].set_color('none')
    ax0.spines['right'].set_color('none')
    ax0.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax0.set_xlabel(xlabel, fontsize="x-large")
    ax0.set_ylabel(ylabel, fontsize="x-large")

    return ax0, ax1, ax2, fig

def plot_tc(times, data1, data2=None, labels1=[], labels2=[], colors1=[], colors2=[],
title1="", title2="", lw=2, fsize="large", xlim=None, ylim=None, xlabel="", ylabel=""):
    """
    plot time series data. Draw single plot if 1 dataset is given and split plot
    if 2 are given. if no colors are specified, grayscale values between 0.3 and 1
    are used.
    """
    if data2:
        ax1, ax2, fig = make_splitplot(xlabel=xlabel, ylabel=ylabel)
        if not labels2:
            labels2 = list(range(1,len(data2)+1))
        if not color2:
            colors2=[(x, x, x) for x in np.linspace(start=0.0, stop=0.8, num=len(data2))]
        for d, l, c in zip(data2, labels2, colors2):
            ax2.plot(times, d, label=l, color=c, lw=lw)
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if not labels1:
            labels1 = list(range(1,len(data1)+1))
        if not colors1:
            colors1=[(x, x, x) for x in np.linspace(start=0.0, stop=0.8, num=len(data1))]
    for d, l, c in zip(data1, labels1, colors1):
        ax1.plot(times, d, label=l, color=c, lw=lw)

    if xlim:
        ax1.set_xlim(xlim[0],xlim[1])
        if data2:
            ax2.set_xlim(xlim[0],xlim[1])
    if ylim:
        ax1.set_ylim(ylim[0],ylim[1])
        if data2:
            ax2.set_ylim(ylim[0],ylim[1])
    if data2:
        ax2.legend()
        ax1.set_xticks([])
        ax2.set_title(title2, fontsize=fsize)
    if not data2:
        ax1.set_xlabel(xlabel, fontsize=fsize)
        ax1.set_ylabel(ylabel, fontsize=fsize)
    ax1.set_title(title1, fontsize=fsize)
    ax1.legend()
    plt.show()

def barplot(data1, data2=None, colors1=[], colors2=[], xticks1=None, xticks2=None, xlabel="", ylabel=""):
    """
    make barplots
    """
    if data2:
        ax1, ax2, fig = make_splitplot(xlabel=xlabel, ylabel=ylabel)
        if not colors2:
            colors2=[(x, x, x) for x in np.linspace(start=0.0, stop=0.8, num=len(data2))]
        ax2.bar(np.array(range(len(data2))), np.mean(data2, axis=1), yerr=np.std(data2, axis=1)/np.sqrt(len(data2[0])), color=colors2, tick_label=xticks2)
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
    if not colors1:
        colors1=[(x, x, x) for x in np.linspace(start=0.0, stop=0.8, num=len(data1))]
    ax1.bar(np.array(range(len(data1))), np.mean(data1, axis=1), yerr=np.std(data1, axis=1)/np.sqrt(len(data1[0])), color=colors1, tick_label=xticks1)
    plt.show()

def plot_response(response):
    # probably needs revision
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
    #probably needs revision
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
