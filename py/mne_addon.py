"""
Functions for working with MNE-python and analysis of EEG and MEG data
"""
from mne.io import read_raw_fif, read_raw_brainvision, RawArray
from mne import read_epochs, grand_average, create_info
from mne.epochs import EpochsFIF
import numpy as np
import os
from os.path import join
import json

def filelist(subjects="all", data_type="eeg", data_kind="raw", fname="Augenmitte.vhdr", blocks=[]):
	"""
	Returns a list of all files for a certain combination of subjects and blocks/conditions
	subjects (str or list of str): can be "all" (default), in which case subject
		names are read from a config file. Else can be a list with a string for each subject
		number, e.g. ["el02", "el07", "el08"]
	data_type ("eeg" or "meg"): needs to be specified since the data structure is different for
		the meg and eeg experiments
	data_kind ("raw" or "processed"): if "raw" (default) the files are taken from RAWDIR if
		"processed" they are taken from EXPDIR
	fname (str): filename which encodes experimental condition. only relevant if data_type is "eeg"
	blocks (list of int): blocks to load. only relevant if data_type is "meg"
	return (list of str): filenames for the specified parameters
	"""
	if data_kind == "raw":
		data_dir = os.environ["RAWDIR"]
	elif data_kind == "processed":
		data_dir = os.environ["EXPDIR"]
	else:
		raise ValueError("ERROR! data_kind must be 'raw' or 'processed'")

	if subjects=="all" and data_type=="eeg":
		list_of_subjects = np.loadtxt(join(os.environ["EXPDIR"],"cfg","list_of_subjects_eeg.txt"), dtype="str")
	elif subjects=="all" and data_type=="meg":
		list_of_subjects = np.loadtxt(join(os.environ["EXPDIR"],"cfg","list_of_subjects_meg.txt"), dtype="str")
	else:
		list_of_subjects = subjects

	file_list=[]
	if data_type=="eeg":
		for subject in list_of_subjects:
			subject_files = os.listdir(join(data_dir, subject))
			for file in subject_files:
				if fname in file:
					file_list.append(join(data_dir,subject,file))

	elif data_type=="meg":
		for subject in list_of_subjects:
			for block in blocks:
				file_list.append(join(data_dir, subject+session, subject+str(block)))
	else:
		raise ValueError("ERROR! data_type must be 'eeg' or 'meg' ")

	return file_list

def load_montage(path): # load montag eso we can do #epochs_ica.set_montage(montage)
    electrodes = np.load(os.path.join(path,os.environ["SUBJECT"]+"_montage.npy"))
    nasion = np.load(os.path.join(path,os.environ["SUBJECT"]+"_nasion.npy"))
    eeg_channels = json.load(open(os.environ["EXPDIR"]+"cfg/eeg_channel_names.cfg"))
    names = list(eeg_channels.values())
    dig_ch_pos = dict(zip(names, electrodes))
    montage = DigMontage(point_names=names, dig_ch_pos=dig_ch_pos)
    montage.plot(show_names=True, kind="topomap")
    return montage

def filt_raw(subject, raw_name, outfile_suffix="_filt", low=1, high=30, data_type="eeg", write=True, overwrite=False):
	"""
	load raw file, apply bandpass filter and save the filtered data.
	subject (str): number of the subject in the experiment, e.g. "el07"
	raw_name (str): name of the raw data file to load, e.g. "el07_condition1.fif"
	outfile_suffix (str): is added to the filename when the data is being saved
	low & high (int): upper and lower edge of the band pass filter. Can be None,
		resulting in only low-/highpass or no filtering at all.
	data_type (str): must be "eeg" or "meg"
	write (bool): save filtered data, defaults to True
	overwrite (bool): overwrite option if a file with that name already exists
		defaults to False
	return: filtered raw data
	"""
	file_in = join(os.environ["RAWDIR"],os.environ["SUBJECT"], raw_name)
	file_out = join(os.environ["EXPDIR"], os.environ["SUBJECT"], raw_name+outfile_suffix)
	if data_type == "eeg":
		raw=read_raw_brainvision(file_in, preload=True).filter(low,high)
	elif data_type == "meg":
		raw=read_raw_fiff(file_in, preload=True).filter(low,high)
	else:
		raise ValueError("ERROR! data_type must be 'eeg' or 'meg' ")
	if write:
		raw.save(file_out, overwrite=overwrite)
	return raw

def get_peaks(list_of_epochs, tmin, tmax):

	"""
	compute the mean across all channels (using the abolute values) and then get
	get the maximum of the mean. This is done for each instance in the
	list_of_epochs (usually each instance corresponds to 1 subject) and event
	within each instance .
	list_of_epochs (list of instace of Epochs): list with epoched data.
		All instances of the list must contain the same events
	tmin & tmax (float): start and stop of the interval that is considered

	return (2d array): peaks in the specified interval. Has a row for each events
		and a column for each element in the list of epochs
	"""

	event_id = list_of_epochs[0].event_id
	nmin = (np.where(list_of_epochs[0].times==tmin)[0][0])#first sample
	nmax = (np.where(list_of_epochs[0].times==tmax)[0][0])#last sample

	all_peaks = np.zeros([len(event_id), len(list_of_epochs)])
	for e in event_id:
		evokeds = [epochs[e].average() for epochs in list_of_epochs]
		peaks=[]
		for evoked in evokeds:
			peaks.append(np.max(np.abs(np.mean(evoked.data[:,nmin:nmax],axis=0))))
		all_peaks[int(e)-1] = peaks
	return all_peaks

def get_global_rms(list_of_epochs):
	"""
	compute the root mean square over all subects and all channels for each event

	list_of_epochs (list of instace of Epochs): list with epoched data.
		All instances of the list must contain the same events

	return (2d array): global rms for each event. Has one row for each event and
		one column for each sample
	"""
	n_samples=len(list_of_epochs[0].average().data[0])
	event_id = list_of_epochs[0].event_id
	rms = np.zeros([len(event_id),n_samples])
	for e in event_id:
		evokeds = [epochs[e].average() for epochs in list_of_epochs]
		evokeds_grand_average = grand_average(evokeds)
		rms[int(e)-1] = np.sqrt(np.mean(np.square(evokeds_grand_average.data), axis=0))

	return rms

def get_chs(raw, ch_type="eeg"):
	"""
	read and return channel names and locations from Raw op Epoch object
	For MEG Sensors(which are described by a vector and a linear transformation)
	the location needs to be transformed from the device to the head coordinate
	system unsing the transformation matrix from the file info. Th first three
	elements in the returned aray describe the position.
	"""
	# info["dig"]=list(filter(lambda point: point["kind"]==4, raw.info["dig"])) # the cap has a reference and a ground that are stored here
    channels = list(filter(lambda ch: ch_type.upper() in ch["ch_name"], raw.info["chs"])) # complete channel info
	ch_locs = [ch["loc"] for ch in channels]  # only location
	ch_names = [ch["ch_name"] for ch in channels]
	if ch_type="meg":
		T = list(raw.info["dev_head_t"].values())[2] #transformation matrix
		for count, loc in zip(range(len(ch_locs)),ch_locs):
			loc = loc.reshape([3, 4], order="F")# reshape into 3 by 4 matrix
			shift = np.expand_dims(loc[:, 0], 1)# add the left column to the right
			loc = np.append(loc,shift, axis=1)# remove it to the left
			loc = np.delete(loc,0, 1)
			loc = np.vstack([loc, [0, 0, 0, 1]])
			loc = np.matmul(np.matmul(T, loc), [0, 0, 0, 1])[0:3]
			ch_locs[count] = loc

	return ch_locs, ch_names

def write_chs(ch_locs, ch_names, fname):
	"""
	create a fake dataset to save channel locs and names
	e.g. after creating a custom electrode layout
	"""
	meg_chs = 0
	eeg_chs = 0
	for ch in ch_names:
		if "MEG" in ch:
			ch_type="meg"
			meg_chs=1
		if "EEG" in ch:
			ch_type="eeg"
			eeg_chs=1
	if meg_chs+eeg_chs !=1:
		raise ValueError("list must contain exactly 1 channel type!")

	info=create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_type)
    for ch, loc in zip(info["chs"],ch_locs):
    	ch["loc"] = loc
    raw = RawArray(data=np.random.randn(len(eeg_channels),100), info = info)
    raw.save(fname, overwrite=False)

def make_fwd(info, trans, subject="",write=True, mindist=2, layers=1):
	"""
	Compute the forward solution. Requires the info from the raw data and the
	trans-file that is created during the coordinate frame alignment
	"""
    if not subject:
        subject = os.environ["SUBJECT"]
    src = setup_source_space(subject=subject, spacing='oct6', add_dist=False)
    surfs = read_bem_surfaces(os.path.join(os.environ["SUBJECTS_DIR"], subject, "bem", subject + "-5120"*layers+"-bem.fif"))
    bem = make_bem_solution(surfs)
    fwd = make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=mindist, n_jobs=2)
    if write:
        write_forward_solution(os.path.join(os.environ["EXPDIR"],subject,subject+".fwd"),fwd)
    return fwd

def source_estimate(epochs, fwd, noise_cov, method="beamformer", data_cov=None, reg=0.05, snr=3.,
pick_ori=None, parc='aparc.a2009s', roi=["G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo"], plot=False):

	"""
	Compute a source estimate for epoched data. Returns the time course of the
	activation for each point in the source space. Values are returned in a
	dictionary with one entry for each event type in the epochs.
	If the method is "beamformer", a spatial filter is computed on the spacified
	region of interest. In that case also a covariance matrix for the data
	needs to be specified. If plot =True, plot the source space with the
	at the time of maximum activation.
	for details on the arugments, check documnetation on the MNE functions:
	 mne.beamformer.make_lcmv & mne.minimum_norm.make_inverse_operator
	"""
	if not type(data) == EpochsFIF:
		raise ValueError("source estimate can be only computed for epoched data!")

	if method == "beamformer":
		print("computing beamformer on evoked data")
		if not data_cov:
			raise ValueError("to compute a beamformer, the data variance must be specified!")
		if roi:
			count=0
			for r in roi: # add all labels
				for hemi in ["lh","rh"]:
					if not count:
						label = read_labels_from_annot(os.environ["SUBJECT"], parc=parc, hemi=hemi, regexp=r)[0]
					else:
						label += read_labels_from_annot(os.environ["SUBJECT"], parc=parc, hemi=hemi, regexp=r)[0]
					count+=1
		else:
			label=None
			print("no region specified, computing beamformer on the whole source space...")
		inverse_operator = make_lcmv(epochs.info, fwd, data_cov, reg=reg, noise_cov=noise_cov, rank=None, label=label)
	else:
		inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov)
		print("computing source estimate using "+method)

	evokeds_stc = dict.fromkeys(events)
	for event in epochs.event_id.keys():
		evoked = epochs[event].average()

		if method=="beamformer":
			evokeds_stc[event] = apply_lcmv(evoked, inverse_operator, max_ori_out='signed')
		else:
			evokeds_stc[event] = apply_inverse(evoked, inverse_operator, lambda2=1. / snr ** 2, method=method)

	if plot:
		for hemi in ["lh","rh"]:
			vertno_max, time_max = stc.get_peak(hemi=hemi)
			brain = stc.plot(hemi=hemi, initial_time=time_max)
			brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue', scale_factor=0.6, alpha=0.5)
			brain.add_text(0.1, 0.9, '%s (plus location of maximal activation)' % method, 'title', font_size=14)
		mlab.show()

	return evokeds_stc

if __name__ == "__main__":
	files = filelist(subjects=["eegl01","eegl03"], data_kind="processed", fname="Augenmitte-epo.fif")
	list_of_epochs=[]
	for f in files:
		list_of_epochs.append(read_epochs(f, preload=False))
	rms=get_global_rms(list_of_epochs)
