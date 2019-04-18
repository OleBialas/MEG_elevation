import os
import sys
sys.path.append(os.environ["PYDIR"])
from mne import find_events, concatenate_raws, pick_types
from mne.epochs import Epochs
from mne.io import read_raw_fif
from matplotlib import pyplot as plt
import numpy as np
from mne.channels import read_montage


subjects=["eegl01","eegl02","eegl03","eegl04","eegl05","eegl06","eegl07","eegl08"]
condition="Augenmitte"
n_blocks=4
montage = read_montage("easycap-M10")
# load data for all subjects in dictionary:
raws=[]
for subject in subjects:
	os.environ["SUBJECT"] = subject
	for i in range(1,n_blocks+1):
		raws.append(read_raw_fif(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_"+str(i)+condition+"_filt.raw")))
raw = concatenate_raws(raws)
raw.info["bads"] =["1","2","64"]
events = find_events(raw)
picks = pick_types(raw.info)
tmin = -0.1
tmax = 1
baseline = (-0.1,0)
reject_thresh = dict(eeg=0.0002)
epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=baseline, reject=reject_thresh, preload=True)