import sys
import os
sys.path.append(os.environ["PYDIR"])
from mne import read_forward_solution, read_cov, read_labels_from_annot
from mne.beamformer import make_lcmv, apply_lcmv
from mne_preprocessing import load_epochs
from matplotlib import pyplot as plt
import numpy as np


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

	if save:
		plt.savefig(os.environ["EXPDIR"]+title)

	plt.show()

	
if __name__ =="__main__":
	subjects = ["el04a","el05a"]
	blocks = dict(supine=["1l","2l","3l"], sedentary=["1s","2s","3s"])
	hemis = ["left", "right", "both"]
	for subject in subjects:
		for condition in blocks.keys():
			for hemi in hemis:
				os.environ["SUBJECT"]=subject
				title = subject+" in %s position, radial component of  AC in %s hemisphere" % (condition, hemi)
				average_stc, times, average_trials, event_id = avg_blocks_stc_data(blocks[condition], hemi=hemi, pick_ori="normal")
				plot_time_series_data(average_stc, times, labels=event_id, stop=0.85, title=title, save=True)