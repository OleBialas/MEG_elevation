import sys
import os
sys.path.append(os.environ["PYDIR"])
from mne_preprocessing import load_epochs
from mne import read_forward_solution, read_cov, stc_to_label, read_labels_from_annot
from mne.beamformer import make_lcmv, apply_lcmv
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_snr_estimate
from matplotlib import pyplot as plt
import numpy as np
import json
cfg = json.load(open(os.path.join(os.environ["EXPDIR"],"cfg","elevation.cfg")))
blocks = cfg["meg_blocks"][:-3]
os.environ["SUBJECT"] = "el04a"
if os.environ["SUBJECT"] == "el01b":
	blocks = ["1","2","3","4","5","6"]

fwd = read_forward_solution(os.path.join(os.environ["DATADIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+".fwd"))
evokeds_stc=[]
n_trials = [0, 0, 0, 0]

# generate label
roi=["G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo"]
labels=[]
for i in roi:
	labels += read_labels_from_annot(os.environ["SUBJECT"][:-1], parc='aparc.a2009s', hemi="both", regexp=i)
label = labels[0]+labels[1]+labels[2]+labels[3]


for block in blocks: # average over all blocks
	
	epochs = load_epochs(block)
	evokeds =[epochs[event].average() for event in epochs.event_id.keys()]
	noise_cov = read_cov(os.path.join(os.environ["DATADIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_noise_cov.fif"))
	data_cov = read_cov(os.path.join(os.environ["DATADIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_data_cov.fif"))
	inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov)
	filters = make_lcmv(epochs.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, rank=None, label=label)
	
	for evoked,count in zip(evokeds,range(len(evokeds))):
		n_trials[count] += evoked.nave
		if block == cfg["meg_blocks"][0]:
			evokeds_stc.append(apply_lcmv(evoked, filters, max_ori_out='signed'))
		else:
			evokeds_stc[count] += apply_lcmv(evoked, filters, max_ori_out='signed').data

fig, ax = plt.subplots(2, sharex=True, sharey=True)
for stc, event in zip(evokeds_stc, epochs.event_id.keys()):
	ax[0].plot(1e3 * stc.times, np.mean(stc.lh_data, axis=0), label=event)
	ax[0].set_title("Left Hemisphere")
	ax[1].plot(1e3 * stc.times, np.mean(stc.rh_data,axis=0), label=event)
	ax[1].set_title("Right Hemisphere")
plt.xlabel('time (ms)')
plt.ylabel('LCMV value')
plt.legend()
plt.show()



# Generate a functional label from source estimates
tmin, tmax = 0.63, 0.66
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2, method, pick_ori="normal",)
stc_mean = stc.copy().crop(tmin, tmax).mean() # Make an STC in the time interval of interest and take the mean

label_name = "transversetemporal-rh"
label = mne.read_labels_from_annot("el04", regexp=label_name)[0]
stc_mean_label = stc_mean.in_label(label)
data = np.abs(stc_mean_label.data)
stc_mean_label.data[data < 0.6 * np.max(data)] = 0.

func_labels, _ = stc_to_label(stc_mean_label, src=src, smooth=True, connected=True)



# hemisphere difference?



vertno_max, time_max = stc.get_peak(hemi='rh')
surfer_kwargs = dict(
    hemi='rh', subjects_dir=os.environ["SUBJECTS_DIR"],
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)


#beacmformer:
filters = make_lcmv(epochs.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori='max-power', rank=None)
stc = apply_lcmv(evoked, filters, max_ori_out='signed')


brain = stc.plot(hemi='lh', views='lat', subjects_dir=os.environ["SUBJ"],
                 initial_time=0.1, time_unit='s', smoothing_steps=5)


"""Check whether SSP projections in data and spatial filter match."""
info = evoked.info
proj_data, _, _ = make_projector(info['projs'], filters['ch_names'])
if not np.allclose(proj_data, filters['proj'],atol=np.finfo(float).eps, rtol=1e-13):
        raise ValueError('The SSP projections present in the data do not match the projections used when calculating the spatial filter.')