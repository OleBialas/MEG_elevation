import os
import json
import sys
sys.path.append(os.environ["EXPDIR"]+"py/")
from mne.io import read_raw_fif
from mne.epochs import Epochs
from mne_erp import evoked_rms, plot_rms_mag_grad
from mne import read_events
import numyp as np
cfg = json.load(open(os.environ["EXPDIR"]+"cfg/elevation.cfg"))
os.environ["SUBJECT"] = "el04a"

for block in cfg["meg_blocks"]:
	raw = read_raw_fif(os.path.join(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+block+".fif"))
	raw.info["bads"] = list(np.loadtxt(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+".bads", dtype=str)) 
	events = read_events(os.path.join(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+block+".eve"))
	epochs = Epochs(raw, events, cfg["epochs"]["event_id"], cfg["epochs"]["time"][0],cfg["epochs"]["time"][1],
		baseline=(cfg["epochs"]["baseline"][0],cfg["epochs"]["baseline"][1]), reject=cfg["epochs"]["reject"])
	evokeds = [epochs[event].average() for event in sorted(cfg["epochs"]["event_id"].keys())]
	rms_mag = evoked_rms(evokeds, cfg["epochs"]["event_id"], ch_type="mag")
	rms_grad = evoked_rms(evokeds, cfg["epochs"]["event_id"], ch_type="grad")
	plot_rms_mag_grad(rms_grad, rms_mag, evokeds[0].times, cfg["epochs"]["event_id"], title=os.environ["SUBJECT"]+block)

#



block = "1"
raw = read_raw_fif(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+".fif", preload=True)
events = read_events(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+".eve")

raw.filter(None , 200)
picks = pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
epochs = Epochs(raw,events,tmin=-0.1,tmax=1.0,baseline=(-0.1,0), preload=True)



#raw.plot()
#raw.info["bads"]=list(np.loadtxt(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+".bads",dtype=str))
picks=pick_types(raw.info)
epochs = Epochs(raw, events, cfg["event_id"], cfg["time"][0],cfg["time"][1], baseline=(cfg["baseline"][0],cfg["baseline"][1]), preload=True)




for i in sorted(cfg["event_id"].keys()):
    epochs[i].average().plot()

plot_evoked_topo(epochs["pos1"].average(),background_color='w')
