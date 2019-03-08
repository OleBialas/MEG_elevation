from mne.io import read_raw_fif
from mne.epochs import Epochs
from mne import read_events, pick_types, pick_channels
from mne.preprocessing import maxwell_filter
from matplotlib import pyplot as plt
import numpy as np
import os
import json
plt.ion()


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

