import os
from mne import find_events, pick_channels
from mne.io import read_raw_fif
import numpy as np
from mne import read_events


def remove_buttons_and_targets(blocks, write=True):
    """
    throw the targets, then throw events away that are corrupted by motor-activity from pressing the button
    """
    for block in blocks:
        events, targets = get_oneback_targets(block)
        events = np.delete(events, targets, axis=0) # remove targets
        i=0
        while i+1<len(events):
            i+=1
            if events[i][2] == 8192 and events[i][0]-events[i-1][0]<1000:
                print("remove event at %s" %(events[i-1][0]))
                events = np.delete(events, i, axis=0)
        if write:
            np.savetxt(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+str(block)+"_cor.eve"), events, fmt="%i")


def get_oneback_targets(block):

    corrects=[]
    false_positives=[]
    false_negatives=[]
    targets=[]
    events=read_events(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+block+".eve"))
    for i in range(1,len(events)):
        if events[i][2] == events[i-1][2] and events[i][2] != 8192: #this is a target
            targets.append(i)
            if events[i+1][2] or events[i+2][2] == 8192:#target was recognized
                corrects.append(i)
            else: #target was not recognized
                false_negatives.append(i)

        if events[i][2] == 8192 and (events[i][0]-events[i-1][0])<1500 and events[i-1][2] != events[i-2][2]:
            # button was pressed and it was not for a target or to end a break
            false_positives.append(i)

    print("of the %s trials in this block, %s were answered correctly. There were %s false positive responses" %(len(targets), len(corrects), len(false_positives)))

    return events, targets



def write_events(blocks):
    """
    Load files for current $SUBJECT for a giveb list of blocks. extract the events
    from the stim channel, recode and write them to a text file
    """

    for block in blocks:
        raw = read_raw_fif(os.path.join(os.environ["EXPDIR"],os.environ["RAWDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+str(block)+".fif"), preload=True)
        events = read_triggers(raw)
        recoded_events = recode_triggers(events)
        recoded_events = recoded_events[recoded_events[:,0].argsort()] #sort array by time
        np.savetxt(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+str(block)+".eve"), recoded_events, fmt="%i")

def read_triggers(raw, bitmasks=[64, 61440], stim_ch="STI101"):

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
    events[:,0] += raw.first_samp


    return events

def recode_triggers(events, ignore=[8192], maxdist=50):

    """
    translates the eventlist where the stimulus condition is encoded by the number
    of subsequent trigger pulses. Pulses are considert to belong to the
    same event if their distance is smaller than maxdist (in samples - defauls to 50).
    Triggercodes listet in ignore are copied into the output array without recoding

    TODO: Last event in the list seems to be missung
    """
    first_of_the_seq=1
    trig_time=0
    trig_counter = 0
    trig_last = 0
    recoded_events=np.array(events[0],ndmin=2)
    for event in events[1:]:
        if event[2] in ignore:
            recoded_events = np.append(recoded_events,np.array(event,ndmin=2), axis=0)
        else:
            if not first_of_the_seq: # still in the sequence
                if event[0] - trig_last < maxdist:
                    trig_counter+=1
                else: # found end of the sequence
                    first_of_the_seq=1
                    recoded_events = np.append(recoded_events,np.array([trig_time,event[1], trig_counter*10],ndmin=2),axis=0)
            if first_of_the_seq:
                trig_time = event[0]
                trig_counter = 1
                first_of_the_seq = 0
        trig_last = event[0]

    return recoded_events

if __name__ == "__main__":
        blocks=["1s", "2s", "3s", "1l", "2l", "3l"]
        write_events(blocks)
