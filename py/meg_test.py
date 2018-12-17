import tdt
import os
import json
import numpy as np
from utilities import prepare_stimuli
import time
LCID = 0x0

def run_experiment():
    RX8 = tdt.initialize_processor(processor="RX8", connection="GB", index=1, path=os.environ["EXPDIR"]+"rpvdsx/play_stereo_preload.rcx")
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    RX8.SetTagVal("n_pulses", cfg["n_pulses"])
    RX8.SetTagVal("isi", cfg["dur_isi"])
    for i in cfg["meg_blocks"]:
        seq = np.loadtxt(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"]+"/lists/meg_test_"+str(i)+".txt")
        input("press enter to start the next block")
        run_block(seq, cfg["speakers"], cfg["multiplexer"],cfg["n_pulses"], RX8, cfg["FS"])

def run_block(seq, speakers, multiplexer, n_pulses, RX8, FS):

    #load all buffers:
    for speaker in speakers:
        left, right = prepare_stimuli(speaker_nr=speaker, adapter=True)
        if speaker == speakers[0]:
            print("setting buffers...")
            RX8.SetTagVal("buflen",len(left))
            RX8.SetTagVal("dur",int(len(left)/FS*1000))
        RX8._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), 'left', 0, left) # = RP2.WriteTagV("left", 0, left)
        RX8._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), 'right', 0, right) # = RP2.WriteTagV("left", 0, left)
        time.sleep(1.0) # wait until buffers are written

    for i in range(len(seq)):

        RX8.SetTagVal("mux",multiplexer[str(int(seq[i]))]) # switch the multiplexer to the current stimulus
        pulse_number = RX8.GetTagVal("cur_n")

        if pulse_number == n_pulses:
            RX8.SoftTrg(1) # start pulsetrain

        while RX8.GetTagVal("playback"):
            print(RX8.GetTagVal("playback"))
            print(i)
            time.sleep(0.01)

        """
        if i != len(seq)-1: # check if this is not the last trial of the block
            if seq[i] != seq[i+1]: # check if this is not a oneback-target
                left, right = prepare_stimuli(speaker_nr=seq[i], adapter=True) # reload stimulus
                RX8._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), str(int(seq[i])) + "_left", 0,
                                         left)  # = RP2.WriteTagV("left", 0, left)
                RX8._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), str(int(seq[i])) + "_right", 0,
                                         right)  # = RP2.WriteTagV("left", 0, left)
                #RX8.WriteTagV(seq[i] + "_left", 0, left)
                #RX8.WriteTagV(str(seq[i]) + "_right", 0, right)
                print("reloading speaker "+str(seq[i]))
        
        while RX8.GetTagVal("cur_n") == pulse_number: # wait until next trial begins
            time.sleep(0.01)
            print("waiting for pulsetrain")
        """

if __name__ == "__main__":
    os.environ["SUBJECT"] = "el03"  # <-- Enter Subject here
    os.environ["EXPDIR"] = "C:/Projects/MEG_Elevation/"
    run_experiment()

