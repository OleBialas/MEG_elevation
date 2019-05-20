import os
import sys
sys.path.append(os.environ["PYDIR"])
from mne import read_epochs
from mne.preprocessing import read_ica
from mne.time_frequency import tfr_morlet, psd_multitaper
from matplotlib import pyplot as plt

os.environ["SUBJECT"] = "eegl03"
epochs = read_epochs(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_Augenmitte-epo.fif"))
ica = read_ica(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"-ica.fif"))
#reject components 0 (blinks), 5(eye-movement), 9,11,14 and 17 (high frequency noise)
epochs_ica = ica.apply(epochs)
epochs_ica.apply_baseline(baseline=(0.5,0.6)) # use 100ms before stimulus onset as baseline
epochs_ica.crop(tmin=0.5,tmax=1.0)
epochs_ica.average().plot()
plt.tight_layout()

#now try wavelet analysis:
epochs.plot_psd(fmin=2., fmax=40.)
freqs = np.logspace(*np.log10([5, 35]), num=10)
n_cycles = freqs / 5.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=3, n_jobs=1)
