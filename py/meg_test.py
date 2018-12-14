import tdt
import os
import json
import numpy as np
from utilities import prepare_stimuli
import time

def run_experiment():
    RX8 = tdt.initialize_processor(processor="RX8", connection="GB", index=2, path=os.environ["EXPDIR"]+"rpvdsx/play_stereo_preload.rcx")
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    for i in cfg["meg_blocks"]:
        seq = np.loadtdt(os.environ["EXPDIR"]+os.environ["SUBJECT"]+"/lists/meg_test_"+str(i)+".txt")
        run_block(seq, cfg["speakers"], cfg["multiplexer"],cfg["n_pulses"], RX8)

def run_block(seq, speakers, multiplexer, n_pulses, RX8):

    #load all buffers:
    for speaker in speakers:
        left, right = prepare_stimuli(speaker_nr=speaker, adapter=True)
        if speaker == speakers[0]:
            RX8.SetTagVal("buflen",len(left))
        RX8.WriteTagV(speaker+"_left", 0, left)
        RX8.WriteTagV(speaker+"_right", 0, right)
    time.sleep(1.0) # wait until buffers are written

    for i in range(len(seq)):

        RX8.SetTagVal("mux",multiplexer[str(seq[i])]) # switch the multiplexer to the current stimulus
        pulse_number = RX8.GetTagVal("cur_n")

        if pulse_number == n_pulses:
            RX8.SoftTrg(1) # start pulsetrain

        while RX8.GetTagVal("playback"):
            time.sleep(0.01)

        if i != len(seq)-1: # check if this is not the last trial of the block
            if seq[i] != seq[i+1]: # check if this is not a oneback-target
                left, right = prepare_stimuli(speaker_nr=seq[i], adapter=True) # reload stimulus
                RX8.WriteTagV(seq[i] + "_left", 0, left)
                RX8.WriteTagV(seq[i] + "_right", 0, right)
                print("reloading speaker "+str(seq[i]))

        while RX8.GetTagVal("cur_n") == pulse_number: # wait until next trial begins
            time.sleep(0.01)





