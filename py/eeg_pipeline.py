import os
import sys
sys.path.append(os.environ["PYDIR"])
from mne import read_epochs
from mne.preprocessing import read_ica

os.environ["SUBJECT"] = "eegl03"
epochs = read_epochs(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_Augenmitte-epo.fif"))
ica = read_ica(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"-ica.fif"))

#reject components 0 (blinks), 5(eye-movement), 9,11,14 and 17 (high frequency noise)
