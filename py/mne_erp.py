from mne.io import read_raw_fif
from mne.epochs import Epochs
from mne import read_events, pick_types, pick_channels
from matplotlib import pyplot as plt
import numpy as np
expdir = "C:/Projects/Elevation/bennewitz/"

def get_epochs(subject, block, event_id, time, baseline,filt=(0.1,200), selection=None, reject={}, exclude_bads=True):

    """
    Load raw data and compute epochs
    :param subject: (str) subject name
    :param block: (int) number of the block (name of the raw file is determined by subject name and block)
    :param event_id: (dict) stimulus conditions with event code
    :param time: (list/tuple) start and stop of epoch in seconds
    :param baseline: (list/tuple) start and stop of baseline in seconds
    :param selection: (list/None) channels included in the epochs, if None all channels are used
    :param reject: (dict) amplitude above which epochs are rejected. Has to be specified for Mags, Grads
    and EEG separately. If empty then no epochs are rejected
    :param filt: (list/tuple) low and high cut off for filter. IF empty, no filter is used
    """
    raw = read_raw_fif(expdir+subject+"/"+subject+str(block)+".fif", preload=True)
    raw.info["bads"] += list(np.loadtxt(expdir+subject+"/bad_channels.txt", dtype=str))
    picks = pick_types(raw.info, selection=selection)
    if filt:
        raw.filter(l_freq=0.1,h_freq=200)
    events = read_events(expdir + subject + "/" + subject + str(block) + "_cor.eve")
    epochs = Epochs(raw, events, event_id=event_id, tmin=time[0], tmax=time[1],
                            baseline=(baseline[0], baseline[1]),picks=picks, reject=reject, preload=True)

    return epochs

def get_evokeds(epochs, event_id, picks=None, exclude=[]):
    """
    Get evoked responses for specified events and channels
    :param epochs: (mne.epochs.Epochs) Epoched data
    :param event_id: (dict) Event name and code for which evoked response will be calculated
    :param picks: ("mag", "grad", list of strings, None) Channels contained in the epoched data. If "mag" or "grad" all
    magneto or gradiometer will be used. If a list of strings is given, evoked potentials will be computed for these channels.
    Defaults to None which means all channels will be used
    :return: list of evokeds
    """
    evokeds=[]
    if picks == "mag":
        picks_ = pick_types(epochs.info, meg="mag")
    elif picks == "grad":
        picks = pick_types(epochs.info, meg="grad")
    if type(picks) == list:
        picks = pick_channels(epochs.info["ch_names"], include=picks)
    for event in sorted(event_id.keys()):
        evokeds.append(epochs[event].average(picks))
    return evokeds

def get_evokeds_rms(evokeds, event_id):
    """
    compute root mean square over all channels for a list of evoked responses
    :param evokeds: (list of mne.Evoked) evoked responses
    :param event_id: (dict) event names and codes
    :return: list containing the root mean square over all channels for each event
    """
    evokeds_rms =[]
    n_channels = len(evokeds[0].data)
    n_samples = len(evokeds[0].data[0])
    for evoked, id in zip(evokeds,sorted(event_id.keys())):
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
