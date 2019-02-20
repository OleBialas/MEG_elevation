import os
from mne import find_events, pick_channels
from mne.io import read_raw_fif
import numpy as np


def write_events(blocks):
    """
    Load files for current $SUBJECT for a giveb list of blocks. extract the events
    from the stim channel, recode and write them to a text file
    """

    for block in blocks:
        raw = read_raw_fif(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+".fif", preload=True)
        events = read_events(raw)
        recoded_events = recode_triggers(events)
        recoded_events = recoded_events[recoded_events[:,0].argsort()] #sort array by time
        np.savetxt(os.environ["DATADIR"]+os.environ["SUBJECT"]+"/"+os.environ["SUBJECT"]+str(block)+".eve", recoded_events, fmt="%i")

def read_events(raw, bitmasks=[64, 61440], stim_ch="STI101"):

    """
    Read trigger from stimulus channel data of raw file. The data is read for each
    element in the bitmasks-array so the returned list is separated by the trigger value
    and needs to be sorted by time at some point.
    """

    picks = pick_channels(raw.info["ch_names"],[stim_ch])
    data = raw.get_data(picks=picks, start=0, stop=None)[0].astype("int32")
    events = np.array([int(raw.info["sfreq"]*raw.times[0]), 0, 0],ndmin=2, dtype="int32")
    for bit in bitmasks:
        bdata = bit&data
        n=0
        while n<len(data):
            curTrg = bdata[n]
            if curTrg != 0:
                idx=n
                while idx<len(data) and bdata[idx]>0:
                    idx+=1
                curTrg = np.median(bdata[n:idx])
                events = np.append(events,np.array([round(raw.info["sfreq"]*raw.times[n]), 0, curTrg],ndmin=2, dtype="int32"),axis=0)
                n=idx
            else:
                n+=1

    return events

def recode_triggers(events, ignore=[8192], maxdist=50):

    """
    translates the eventlist where the stimulus condition is encoded by the number
    of subsequent trigger pulses. Pulses are considert to belong to the
    same event if their distance is smaller than maxdist (in samples - defauls to 50).
    Triggercodes listet in ignore are copied into the output array without recoding
    """
    first_of_the_seq=1
    trig_time=0
    trig_counter = 0
    trig_last = 0
    for event in events:
        if np.sum(event==events[0])==3:
            recoded_events=np.array(event,ndmin=2)
        elif event[2] in ignore:
            recoded_events = np.append(recoded_events,np.array(event,ndmin=2), axis=0)
        else:
            if not first_of_the_seq: # still in the sequence
                if event[0] - trig_last < maxdist:
                    trig_counter+=1
                else: # found end of the sequence
                    first_of_the_seq=1
                    recoded_events = np.append(recoded_events,np.array([trig_time,event[1], trig_counter*10],ndmin=2),axis=0)
            elif first_of_the_seq:
                trig_time = event[0]
                trig_counter = 1
                first_of_the_seq = 0
        trig_last = event[0]

    return recoded_events

if __name__ == "__main__":
        os.environ["SUBJECT"] = "el05a"
        blocks=["1s","2s","3s","1l","2l","3l"]
        write_events(blocks)
