import os
import json
from mne.io import read_raw_fif
from mne import pick_types, read_events, pick_channels
from mne.epochs import Epochs
from mne.viz import plot_evoked_topo

os.environ["SUBJECT"] = "el99p" # <-- Enter Subject here

cfg = json.load(open(os.environ["EXPDIR"]+"cfg/epochs.cfg"))

block = 1
# STEP 1: Plot raw data and identify bad channels
raw = read_raw_fif(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+"_raw.fif", preload=True)
events = read_events(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+"_cor.eve")
raw.filter(0.1 , 200)
raw.plot()

#negative selection
raw.info["bads"]=["MEG2141","MEG2142","MEG2143","MEG1921"]
picks=pick_types(raw.info)
epochs = Epochs(raw, events, cfg["event_id"], cfg["time"][0], cfg["time"][1], baseline=(cfg["baseline"][0],cfg["baseline"][1]), picks=picks, preload=True)


#positive selection:
good_chs = ["MEG0233", "MEG1621", "MEG0231", "MEG0212", "MEG0243", "MEG1643", "MEG0222", "MEG1613"]
picks = pick_channels(raw.info["ch_names"],include=good_chs)
epochs = Epochs(raw, events, cfg["event_id"], cfg["time"][0], cfg["time"][1], baseline=None, preload=True, reject=cfg["reject"], picks=picks)


for i in sorted(cfg["event_id"].keys()):
    epochs[i].average().plot()

plot_evoked_topo(epochs["pos1"].average(),background_color='w')
