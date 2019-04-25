from mne.io import read_raw_fif
from mne.epochs import Epochs
from mne import read_events, pick_types, pick_channels, find_events, concatenate_raws
from mne.preprocessing import maxwell_filter
from matplotlib import pyplot as plt
import numpy as np
import os
import json
plt.ion()

def eeg_evoked_grand_avg(subjects, condition, include_chs=[], event_id=None):

    """
    Load and concatenate filtered raw data and compute evoked response
    returns evoked response for all events in the data_cov

    Parameters:
    subjects (list of str): names of the subjects to include, if empty, list is read from a text file containing all subjects
    condition (str): name of the experimental condition
    event_id (list of str | None): event types to include in the epoched data, if None (=default) take all events
    include_chs (list of str): names of the channels to include, if empty (=default) include all channels
    """
    if not subjects:
        subjects = np.loadtxt(os.environ["EXPDIR"] + "/cfg/eeg_subjects.cfg", dtype="str")
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation_eeg.cfg"))
    rms_all_subjects = { subject: list([]) for subject in subjects}
    for subject in subjects:
    	os.environ["SUBJECT"] = subject
    	raws=[]
    	for i in range(1,cfg["blocks_per_condition"]+1):
    		raws.append(read_raw_fif(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_"+str(i)+condition+"_filt.raw")))
    raw = concatenate_raws(raws)
    events = find_events(raw)
    picks = pick_channels(raw.info["ch_names"],include=include_chs) # Fz =5, Fc3 = 43, Fc4=44, Cz=14, Pz=25
    epochs = Epochs(raw, events, event_id, tmin=cfg["epochs"]["time"][0], tmax=cfg["epochs"]["time"][1],
                baseline=tuple(cfg["epochs"]["baseline"]),picks=picks, reject=cfg["epochs"]["reject"], preload=True)
    return epochs

def evoked_rms(evokeds, event_id, ch_type="mag"):

    evokeds_rms = []
    n_channels = len(evokeds[0].data)
    n_samples = len(evokeds[0].data[0])
    for evoked in evokeds:
        ch_rms = np.zeros(n_samples)
        for ch in evoked.pick_types(meg="mag").data:
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
