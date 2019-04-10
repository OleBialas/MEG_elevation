import sys
import os
sys.path.append(os.environ["PYDIR"])
from mne import read_forward_solution, read_cov, read_labels_from_annot
from mne.beamformer import make_lcmv, apply_lcmv
from mne_preprocessing import load_epochs
from matplotlib import pyplot as plt
from mne.minimum_norm import make_inverse_operator, apply_inverse
import numpy as np
from mayavi import mlab
import json
cfg = json.load(open(os.path.join(os.environ["EXPDIR"],"cfg","elevation.cfg")))


def avg_blocks_stc_data(blocks, hemi="both", reg=0.05, pick_ori=None, parc='aparc.a2009s', roi=["G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo"]):
	
	"""
	Load Source estimate data in region of interest for each block, using the function beamformer an average over all blocks
	blocks(list of str): acronyms of the blocks to average over. Have to be the same for all participants
	"""
	all_blocks=dict.fromkeys(blocks)
	all_blocks["n_trials"] = dict.fromkeys(blocks)
	for block in blocks:
		stc, event_id, n_trials = beamformer(block, reg, pick_ori, parc, roi)
		all_blocks[block]=stc
		all_blocks["n_trials"][block] = n_trials
		
	average_stc = np.zeros([len(event_id),len(stc[0].times)])
	average_trials = np.zeros([4],dtype=int)
	for i in range(len(average_stc)):
		for block in blocks:
			if hemi == "both":
				average_stc[i] += np.average(all_blocks[block][i].data, axis=0)
			elif hemi == "left":
				average_stc[i] += np.average(all_blocks[block][i].lh_data, axis=0)
			elif hemi == "right":
				average_stc[i] += np.average(all_blocks[block][i].rh_data, axis=0)
		average_stc[i] /= len(blocks)
	for block in blocks:
		average_trials += all_blocks["n_trials"][block]

	return average_stc, stc[0].times, average_trials, event_id

def beamformer(block, reg=0.05, pick_ori=None, parc='aparc.a2009s', roi=["G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo"]):
	
	"""
	Load Evoked data for given block, and apply LCMV spatrial filter

	parc (str): type of parcellation that is used, can be ‘aparc’ or ‘aparc.a2009s’.
	roi (list of str): regions of the parcellation used that the spatial filter is computed on
	reg (float): The regularization for the whitened data covariance.
	pick_ori (None | ‘normal’ | ‘max-power’ | ‘vector’): see MNE-documnetation
	"""

	try:
		fwd = read_forward_solution(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+".fwd"))
		noise_cov = read_cov(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_noise_cov.fif"))
		data_cov = read_cov(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_data_cov.fif"))
	except FileNotFoundError:
		print("Error! Forward solution and covariance matrices for noise and data must be computed before running this ananlysis")
		return
	
	count=0
	for r in roi: # add all labels
		for hemi in ["lh","rh"]:
			if not count:
				label = read_labels_from_annot(os.environ["SUBJECT"][:-1], parc=parc, hemi=hemi, regexp=r)[0]
			else:
				label += read_labels_from_annot(os.environ["SUBJECT"][:-1], parc=parc, hemi=hemi, regexp=r)[0]
			count+=1

	epochs = load_epochs(block)
	evokeds =[epochs[event].average() for event in epochs.event_id.keys()]
	spatial_filter = make_lcmv(epochs.info, fwd, data_cov, reg=reg, noise_cov=noise_cov, rank=None, label=label)
	stcs, n_trials = [], []		
	for evoked in evokeds:
		n_trials.append(evoked.nave)
		stcs.append(apply_lcmv(evoked, spatial_filter, max_ori_out='signed'))

	return stcs, epochs.event_id.keys(), n_trials

def source_estimate(blocks, method="dSPM", snr=3., plot=True):

	try:
		fwd = read_forward_solution(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+".fwd"))
	except:
		print("Forward solution must be computed before running this ananlysis")
	for block in blocks: # average over all blocks
		epochs = load_epochs(block)
		evokeds =[epochs[event].average() for event in epochs.event_id.keys()]
		noise_cov = read_cov(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_noise_cov.fif"))
		inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov)
		count=0
		for evoked in evokeds:
			if block == blocks[0] and count == 0:
				stc = apply_inverse(evoked, inverse_operator, lambda2=1. / snr ** 2, method=method)
			else:
				stc.data += apply_inverse(evoked, inverse_operator, lambda2=1. / snr ** 2, method=method).data
		stc.data /= (len(blocks)*len(epochs.event_id)) #divide by number of blocks x number of epochs
		print("divide by "+str((len(blocks)+len(epochs.event_id))))
	
	if plot:
		for hemi in ["lh","rh"]:
			vertno_max, time_max = stc.get_peak(hemi=hemi)
			brain = stc.plot(hemi=hemi, initial_time=time_max)
			brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue', scale_factor=0.6, alpha=0.5)
			brain.add_text(0.1, 0.9, '%s (plus location of maximal activation)' % method, 'title', font_size=14)
			brain.save_image(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_%s_.jpg" % method))
		mlab.show()
	
	return stc


def plot_time_series_data(data, times, start=None, stop=None, labels=None, title=None, ylim=None, save=False):
	"""
	"""
	if stop:
		n_stop = np.where(times==stop)[0][0]
	else:
		n_stop = len(times)
	if start:
		n_start = np.where(times==start)[0][0]
	else:
		n_start = 0

	times = times-(cfg["dur_adapter"]+cfg["dur_ramp"]) # subtract adapter and ramp so 0 corresponds to the stimulus onset
	if not labels:
		labels = range(len(data))
	for i, label in zip(data,labels):
		plt.plot(times[n_start:n_stop], i[n_start:n_stop], label=label)
	if title:
		plt.title(title)
	if ylim:
		plt.xlim(ylim)
	if labels:
		plt.legend()

	plt.show()
	if save:
		plt.savefig(os.environ["EXPDIR"]+title)

def stc_both_hemis(blocks, stop=None):

	fig = plt.figure()
	ax0 = fig.add_subplot(111)    # The big subplot
	ax1 = fig.add_subplot(211)
	#ax1.set_ylim(0.05,0.3)
	ax2 = fig.add_subplot(212)
	#ax2.set_ylim(0.05,0.3)
	# Turn off axis lines and ticks of the big subplot
	ax0.spines['top'].set_color('none')
	ax0.spines['bottom'].set_color('none')
	ax0.spines['left'].set_color('none')
	ax0.spines['right'].set_color('none')
	ax0.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	ax0.set_xlabel('time (ms)')
	ax0.set_ylabel('LCMV value')
	for hemi, ax in zip(["left","right"],[ax1,ax2]):
		average_stc, times, average_trials, event_id = avg_blocks_stc_data(blocks, hemi=hemi, pick_ori="normal")
		
		if stop:
			n_stop = np.where(times==stop)[0][0]
		else:
			n_stop = len(times)

		times = times-(cfg["dur_adapter"]+cfg["dur_ramp"])
		for stc, event in zip(average_stc, cfg["epochs"]["event_id"].keys()):
			ax.plot(1e3 * times[0:n_stop], stc[0:n_stop], label=event)
		ax.set_title("%s hemisphere" %(hemi))
		ax.legend()
	plt.show()


if __name__ =="__main__":
	blocks = ["1s","2s","3s"]
	os.environ["SUBJECT"]="el05a"
	source_estimate(blocks)