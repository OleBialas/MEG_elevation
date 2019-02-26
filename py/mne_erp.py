from mne.io import read_raw_fif
from mne.epochs import Epochs
from mne import read_events, pick_types, pick_channels
from mne.preprocessing import maxwell_filter
from matplotlib import pyplot as plt
import numpy as np
import os
import json


def get_epochs(raws, blocks, selection=None, reject=None, exclude_bads=True, filt=True):

    cfg = json.load(open(os.environ["EXPDIR"]+"cfg/epochs.cfg"))
    epochs=[]
    for raw, block in zip(raws,blocks):
        if exclude_bads:
            raw.info["bads"] = list(np.loadtxt(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+".bads", dtype=str))
        if filt:
            raw.filter(None,200)
        picks = pick_types(raw.info, selection=selection)
        if reject:
            reject=cfg["reject"]
        events = read_events(os.environ["DATADIR"] + os.environ["SUBJECT"] + "/" + os.environ["SUBJECT"] + str(block) + ".eve")
        epochs.append(Epochs(raw, events, event_id=cfg["event_id"], tmin=cfg["time"][0], tmax=cfg["time"][1],
                                baseline=(cfg["baseline"][0], cfg["baseline"][1]),picks=picks, reject=reject, preload=True))

    return epochs

def get_evokeds(epochs, event_id, picks=None, exclude=[]):

    evokeds=[]

    if picks == "mag":
        picks = pick_types(epochs.info, meg="mag")
    elif picks == "grad":
        picks = pick_types(epochs.info, meg="grad")
    if type(picks) == list:
        picks = pick_channels(epochs.info["ch_names"], include=picks)
    for event in sorted(event_id.keys()):
        evokeds.append(epochs[event].average(picks))
    return evokeds

def evoked_rms(evokeds, event_id):

    evokeds_rms = []
    n_channels = len(evokeds[0].data)
    n_samples = len(evokeds[0].data[0])
    for evoked in evokeds:
        ch_rms = np.zeros(n_samples)
        for ch in evoked.data:
            ch_rms += np.square(ch)
        ch_rms = np.sqrt((ch_rms/n_channels))
        evokeds_rms.append(ch_rms)
    return evokeds_rms

def plot_rms_mag_grad(data_grad, data_mag, time, event_id, title=""):
    """
    plot rms for gradio and magnetometer
    """

    fig, ax = plt.subplots(2, sharex=True)
    if title:
        fig.suptitle(title)
    for grad, mag, event in zip(data_grad, data_mag, sorted(event_id.keys())):
        ax[0].plot(time, grad, label=event)
        ax[0].set_title("Gradiometer")
        ax[1].plot(time, mag, label=event)
        ax[1].set_title("Magnetometer")
    plt.legend()
    plt.show()


if __name__ =="__main__":

    bads = ["MEG1241", "MEG1831", "MEG2641", "MEG2631"]
    # channels corresponding to auditory areas:
    grads = ["MEG1512", "MEG1513", "MEG0242", "MEG0243", "MEG1612", "MEG1613", "MEG1522", "MEG1523"]
    mags=["MEG0141", "MEG1511", "MEG1541", "MEG0241","MEG0231", "MEG0441", "MEG1611", "MEG1621", "MEG1811"]
    subject="el01b"
    event_id = {"pos1":10, "pos2":20, "pos3":30, "pos4":40}
    blocks = [1,2,3]
    time=(0.61,1)
    baseline=(0.9,1)
    reject=dict(grad=4000e-13, mag = 4e-12)

    for block in blocks:
        epochs = get_epochs(subject, block, event_id, time, baseline, bads, reject, filt=True)
        times = epochs[0].average().times
        evokeds_mag = get_evokeds(epochs, event_id, picks=mags)
        evokeds_grad = get_evokeds(epochs, event_id, picks=grads)
        evokeds_rms_mag = get_evokeds_rms(evokeds_mag, event_id)
        evokeds_rms_grad = get_evokeds_rms(evokeds_grad, event_id)
        plot_rms_mag_grad(data_grad=evokeds_rms_grad, data_mag=evokeds_rms_mag,
                          time = times, event_id= event_id, title="Block "+str(block))
