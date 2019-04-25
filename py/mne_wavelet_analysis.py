import sys
import os
sys.path.append(os.environ["PYDIR"])
import numpy as np
from matplotlib import pyplot as plt
from mne import create_info, EpochsArray, concatenate_raws, find_events, Epochs, pick_channels
from mne.baseline import rescale
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet)
from mne_erp import eeg_evoked_grand_avg
from mne.io import read_raw_fif

subjects=["eegl01","eegl02","eegl03","eegl04","eegl06","eegl07","eegl08","eegl09","eegl11","eegl12","eegl13","eegl14","eegl15","eegl16","eegl17","eegl18","eegl19","eegl20","eegl21"]
for subject in subjects:
    os.environ["SUBJECT"] = subject
    condition="Augenmitte"
    raw = read_raw_fif(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_1Augenmitte_filt.raw"), preload=False)
    picks = [raw.info['ch_names'].index('5'), ] # =Fz
    epochs = Epochs(raw, find_events(raw),picks=picks, tmin=-0.1, tmax=0.9, baseline=(-0.1,0), reject=dict(eeg=60e-6))
    epochs.average().plot()



epochs.plot_psd(fmin=2., fmax=40.)

#TFR using morelt wavelets
freqs = np.arange(2, 30, 1)  # frequencies of interest
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=1, return_itc=True, decim=3, n_jobs=1)
power.plot()
