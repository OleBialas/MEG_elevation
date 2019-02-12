import os
from mne import find_events
from mne.io import read_raw_fif
import numpy as np
import json

def recode_triggers(blocks):
    cfg = json.load(open(os.environ["EXPDIR"]+"cfg/epochs.cfg"))
    for block in blocks:
        raw = read_raw_fif(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+"_raw.fif", preload=True)
        events= find_events(raw)
        recoded_events=[]
        first_of_the_seq=1
        trig_time=0
        trig_counter = 0
        trig_last = 0
        for event in events:
            if event[2] == 8192: # do nothing if buttonpress
                recoded_events.append([event[0],event[2]])
            else:
                if not first_of_the_seq: # still in the sequence
                    if event[0] - trig_last < 50:
                        trig_counter+=1
                    else: # found end of the sequence
                        first_of_the_seq=1
                        recoded_events.append([trig_time, trig_counter*10])
                if first_of_the_seq:
                    trig_time = event[0]
                    trig_counter = 1
                    first_of_the_seq = 0
            trig_last = event[0]
        np.savetxt(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+".eve", np.asarray(recoded_events),fmt='%i', delimiter=",")
