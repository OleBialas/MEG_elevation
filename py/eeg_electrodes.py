from mne.io import read_raw_fif, RawArray
from mne import create_info
import os
import numpy as np

caps=["ac01a","ac02a"]
sizes = ["54", "56"]
reps = 2
blocks = 3


for cap, size in zip(caps, sizes):
	for r in range(1,reps+1):
		for b in range(1,blocks+1):
			raw = read_raw_fif(os.path.join(os.environ["RAWDIR"],cap,cap+str(b*r)+"_size"+size+".fif"))
			eeg_channels = list(filter(lambda ch: "EEG" in ch["ch_name"], raw.info["chs"])) # all eeg channels
			info=create_info(ch_names=[ch["ch_name"] for ch in eeg_channels], sfreq=1000, ch_types="eeg")
			locs = np.array([ch["loc"] for ch in eeg_channels])  #positions of eeg channels
			if r == 1 and b==1:
				locs = np.array(eeg_locs)
			else:
				locs+= np.array(eeg_locs)
	locs /= reps*blocks
	for ch, loc in zip(info["chs"],locs):
		ch["loc"] = loc
	raw = RawArray(data=np.random.randn(len(eeg_channels),100), info = info)
	raw.save(os.environ["RAWDIR"]+"acticap_64_ch_size_"+size+".fif")
