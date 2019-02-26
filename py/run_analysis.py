import os
import json
import sys
sys.path.append(os.environ["EXPDIR"]+"py/")
from mne_erp import *
from mne_preprocessing import *
cfg = json.load(open(os.environ["EXPDIR"]+"cfg/elevation.cfg"))
os.environ["SUBJECT"] = "el04a"


raws = load_raws(cfg["meg_blocks"])
raws_sss = maxwell_filt(raws)

epochs = get_epochs(cfg["meg_blocks"], reject=True, exclude_bads=True, filt=True)

#concatenate epochs:



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
