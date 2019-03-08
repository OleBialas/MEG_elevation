import sys
import os
sys.path.append(os.environ["PYDIR"])
from mne_preprocessing import load_epochs
from mne import read_forward_solution, read_cov, read_labels_from_annot
from mne.beamformer import make_lcmv, apply_lcmv
from mne.minimum_norm import make_inverse_operator, apply_inverse
from matplotlib import pyplot as plt
import numpy as np
import json
from mayavi import mlab
cfg = json.load(open(os.path.join(os.environ["EXPDIR"],"cfg","elevation.cfg")))

"""
TODO:	-Check the configuration for computing beamformer and inverse solution
"""

def average_over_subjects(subjects, blocks, stop=None):
	#problem: unterschiedlich viele vertices in jedem probanden

	average = dict(lh=[], rh=[])
	for subject in subjects:
		os.environ["SUBJECT"] = subject
		stc, n_trials = beamformer(blocks, plot=False)
		times = stc[0].times

		for i in range(len(stc)):
			if subject == subjects[0]:
				average["lh"].append(np.mean(stc[i].lh_data, axis=0))
				average["rh"].append(np.mean(stc[i].rh_data, axis=0))
			else:
				average["lh"] += np.mean(stc[i].lh_data, axis=0)
				average["rh"] += np.mean(stc[i].rh_data, axis=0)
		average["lh"] /= len(subjects)
		average["rh"] /= len(subjects)


	if stop:
		n_stop = np.where(times==stop)[0][0]
	else:
		n_stop = len(times)

	fig = plt.figure()
	ax = fig.add_subplot(111)    # The big subplot
	ax1 = fig.add_subplot(211)
	ax1.set_ylim(0.05,0.65)
	ax2 = fig.add_subplot(212)
	ax2.set_ylim(0.05,0.65)
	# Turn off axis lines and ticks of the big subplot
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

	for lh, rh, event in zip(average["lh"], average["rh"], cfg["epochs"]["event_id"].keys()):
		ax1.plot(1e3 * times[0:n_stop], lh[0:n_stop], label=event)
		ax2.plot(1e3 * times[0:n_stop], rh[0:n_stop], label=event)
	ax1.set_title("mean activity in auditory areas left hemisphere")
	ax2.set_title("mean activity in auditory areas right hemisphere")
	ax.set_xlabel('time (ms)')
	ax.set_ylabel('LCMV value')
	ax1.legend()
	ax2.legend()
	plt.show()


def beamformer(blocks, parc='aparc.a2009s', roi=["G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo"], plot=True):

	try:
		fwd = read_forward_solution(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+".fwd"))
	except:
		print("Forward solution must be computed before running this ananlysis")
	
	count=0
	for r in roi: # add all labels
		for hemi in ["lh","rh"]:
			if not count:
				label = read_labels_from_annot(os.environ["SUBJECT"][:-1], parc=parc, hemi=hemi, regexp=r)[0]
			else:
				label += read_labels_from_annot(os.environ["SUBJECT"][:-1], parc=parc, hemi=hemi, regexp=r)[0]
			count+=1

	evokeds_stc=[]
	n_trials = [0, 0, 0, 0]
	for block in blocks: # average over all blocks
		
		epochs = load_epochs(block)
		evokeds =[epochs[event].average() for event in epochs.event_id.keys()]
		noise_cov = read_cov(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_noise_cov.fif"))
		data_cov = read_cov(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+"_data_cov.fif"))
		filters = make_lcmv(epochs.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, rank=None, label=label)
		
		for evoked,count in zip(evokeds,range(len(evokeds))):
			n_trials[count] += evoked.nave
			if block == blocks[0]:
				evokeds_stc.append(apply_lcmv(evoked, filters, max_ori_out='signed'))
				print("appending")
			else:
				evokeds_stc[count] += apply_lcmv(evoked, filters, max_ori_out='signed').data
				print("adding")
		for stc in evokeds_stc:
			stc.data /= len(blocks) #divide by number of blocks

	if plot:
		plot_source_time_course(evokeds_stc, epochs.event_id, mean=True)

	return evokeds_stc, n_trials

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
				stc += apply_inverse(evoked, inverse_operator, lambda2=1. / snr ** 2, method=method).data
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

def plot_source_time_course(stc_list, event_id, mean=False):
	if mean:
		fig, ax = plt.subplots(2, sharex=True, sharey=True)
		for stc, event in zip(stc_list, event_id.keys()):
			ax[0].plot(1e3 * stc.times, np.mean(stc.lh_data, axis=0), label=event)
			ax[1].plot(1e3 * stc.times, np.mean(stc.rh_data, axis=0), label=event)
			ax[0].set_title("Left Hemisphere mean ERP per condition")
			ax[1].set_title("Right Hemisphere mean ERP per condition")
			plt.legend()

	fig, ax = plt.subplots(4,2, sharex=True, sharey=True)
	for stc, event, count in zip(stc_list, event_id.keys(), range(len(event_id))):
		ax[count,0].plot(1e3 * stc.times, stc.lh_data.T)
		ax[count,1].plot(1e3 * stc.times, stc.rh_data.T)
		ax[count,0].set_title("Left Hemisphere "+event)
		ax[count,1].set_title("Right Hemisphere "+event)
	plt.xlabel('time (ms)')
	plt.ylabel('LCMV value')
	plt.show()


if __name__ == "__main__":

	subjects=["el04a","el05a"]
	blocks=["1l","2l","3l"]
	average_over_subjects(subjects, blocks)