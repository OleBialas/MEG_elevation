from mne.io import read_raw_fif, read_raw_brainvision
from mne import concatenate_raws, read_events, pick_types
from mne.epochs import Epochs
from mne import compute_covariance, write_cov, compute_proj_raw
import numpy as np
import os
import json
cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))

def filt_eeg(subjects, condition):

	#load EEG-data filter and save
	for subject in subjects:
		os.environ["SUBJECT"] = subject
		blocks=[]
		files = os.listdir(os.path.join(os.environ["RAWDIR"],os.environ["SUBJECT"]))
		for file in files:
			if condition+".vhdr" in file:
				blocks.append(file)
		count=1
		for block in blocks:
			raw=read_raw_brainvision(os.path.join(os.environ["EXPDIR"],os.environ["RAWDIR"],os.environ["SUBJECT"],block), preload=True).filter(1,30)
			raw.save(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_"+str(count)+condition+"_filt.raw"), overwrite=True)
			count+=1

def load_raw(block, filt=True, reject_bads=True, write=True):

	raw = read_raw_fif(os.path.join(os.environ["RAWDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+str(block)+".fif"), preload=True)
	if filt:
		raw.filter(None,200)
	if reject_bads:
		raw.info["bads"] = list(np.loadtxt(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+".bads"), dtype=str))
		raw.pick_types()
		raw.info.normalize_proj()
		raw.save(os.path.join(os.environ["EXPDIR"], os.environ["SUBJECT"],os.environ["SUBJECT"]+str(block)+"_filt.fif"))
	return raw

def load_epochs(block, filt=True, reject_bads=True):

	raw = read_raw_fif(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+str(block)+"_filt.fif"), preload=True)
	events=read_events(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_cor.eve"))
	epochs = Epochs(raw, events, cfg["epochs"]["event_id"], cfg["epochs"]["time"][0],cfg["epochs"]["time"][1],
		baseline=(cfg["epochs"]["baseline"][0],cfg["epochs"]["baseline"][1]), reject=cfg["epochs"]["reject"])
	del raw
	return epochs


def write_covariance(method="shrunk"):

	for block in cfg["meg_blocks"]:
		epochs = load_epochs(block)
		noise_cov = compute_covariance(epochs, tmin=cfg["epochs"]["baseline"][0], tmax=cfg["epochs"]["baseline"][1], method="shrunk")
		data_cov = compute_covariance(epochs, tmin=cfg["data_interval"][0], tmax=cfg["data_interval"][1], method="shrunk")
		write_cov(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_noise_cov.fif"),noise_cov)
		write_cov(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_data_cov.fif"),data_cov)

def maxfilt(block, write=True, destination_block="1s"):

	raw = load_raw(block)
	destination = os.environ["RAWDIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+destination_block+".fif"
	raws_sss = (maxwell_filter(raw, destination=destination))
	if write:
		raws_sss.save(os.environ["RAWDIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+"_maxfilt.fif")

	return raws_sss

def load_concatenated_raws(blocks):

	filename = os.path.join(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"])
	raws = [read_raw_fif(filename+block+"_maxfilt.fif", verbose='error') for block in blocks]
	events_list = [read_events(filename+block+".eve") for block in blocks]
	raw, events = concatenate_raws(raws, events_list=events_list)
	return raw

if __name__ =="__main__":
	#weird stuff happens for subject 10
	subjects = ["eegl09","eegl11","eegl12","eegl13","eegl14","eegl15","eegl16","eegl17","eegl18","eegl19","eegl20","eegl21"]
	filt_eeg(subjects, condition="Augenmitte")
