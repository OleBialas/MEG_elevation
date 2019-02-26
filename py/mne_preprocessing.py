from mne.io import read_raw_fif
from mne.preprocessing import maxwell_filter
import numpy as np
import os


def load_raws(blocks, filt=True, mark_bads=True):
    raws=[]
    for block in blocks:
        raw = read_raw_fif(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+".fif", preload=True)
        if filt:
            raw.filter(None,200)
        if mark_bads:
        	raw.info["bads"] = list(np.loadtxt(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+".bads", dtype=str))
        raws.append(raw)
    return raws

def maxwell_filt(raws):

	destination = os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+"1s"+".fif"
	raws_sss = []
	for raw in raws:
		raws_sss.append(maxwell_filter(raw, destination=destination))

	return raws_sss
