from mne.io import read_raw_fif
import os

subjects=["ac01a","ac02a"]
sizes = ["54", "56"]
reps = 2
blocks = 3

for subject, size in zip(subjects, sizes):
	for r in range(1,reps+1):
		for b in range(1,blocks+1):
			raw = read_raw_fif(os.path.join(os.environ["RAWDIR"],subject,subject+str(b*r)+"_size"+size+".fif"))