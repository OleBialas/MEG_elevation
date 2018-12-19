"""
main script for running the elevation experiment.
previously generated stimuli and sequence are loaded dependent on input strings "subject" and "position"
"""

import tdt
import time
import numpy as np
import os
import json
from utilities import prepare_stimuli
LCID = 0x0


def run_experiment():

    cfg = json.load(open(os.environ["EXPDIR"]+"/cfg/elevation.cfg"))
    RX8 = tdt.initialize_processor(processor="RX8", connection="USB", index=1, path=os.environ["EXPDIR"]+"rpvdsx/play_stereo_meg.rcx")  # initialize processor
    for i in cfg["meg_blocks"]:
        input("press Enter to start block nr. "+str(i))
        seq = np.loadtxt(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"]+"/lists/meg_test_block_"+str(i)+".txt")
        run_block(seq, RX8, cfg["trg_codes"], cfg["speakers"])

def run_block( seq, RX8, trg_codes, speakers):

    count = 0
    for i in seq:
        left, right = prepare_stimuli(str(int(i)), adapter=True)
        if count == 0:
            RX8.SetTagVal("playbuflen",len(left))
        RX8.SetTagVal("trg_nr", trg_codes[str(int(i))]) # set the number of trigger pulses send
        RX8._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), 'left', 0, left) # = RP2.WriteTagV("left", 0, left)
        RX8._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), 'right', 0, right) # = RP2.WriteTagV("left", 0, left)
        tic = time.time()
        time.sleep(0.33) # wait while buffering
        RX8.SoftTrg(1) # start trial
        # 247 = inaktiv, 255 = aktiv
        while RX8.GetTagVal("playback"):
            time.sleep(0.01)
        print(time.time()-tic)

        #take a break each 10 trials if we are not currently in a target
        if count >0 and count%10 == 0:
            while RX8.GetTagVal("button") != 255:
                time.sleep(0.01)
        count+=1


if __name__ == "__main__":
    os.environ["SUBJECT"] = "el03"  # <-- Enter Subject here
    os.environ["EXPDIR"] = "C:/Projects/MEG_Elevation/"
    run_experiment()
