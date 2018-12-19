import tdt
import time
import numpy as np
from freefield_table import freefield_table
from utilities import prepare_stimuli
import os
import json

LCID = 0x0

def free_field():
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    seq = np.loadtxt(os.environ["EXPDIR"]+os.environ["SUBJECT"]+"/lists/freefield.txt")
    RX8=[]
    for i in [1,2]:
        RX8.append(tdt.initialize_processor(processor="RX8",connection="GB",index=i, path=os.environ["EXPDIR"]+"rpvdsx/play_noise.rcx"))
    RP2 = tdt.initialize_processor(processor="RP2",connection="GB",index=1, path=os.environ["EXPDIR"]+"rpvdsx/play_stereo.rcx")
    ZB = tdt.initialize_zbus(connection="GB")

    for processor in RX8:
        processor.SetTagVal("n_samples", int(cfg["dur_freefield"]*cfg["FS"]))

    response = np.zeros([len(seq),3]) #output array with three columns: stimulus, response and time
    count = 0
    for i in seq:
        s = freefield_table(Ongoing=[str(int(i))])
        RX8[int(s["RX8"][0]) - 1].SetTagVal("ch_nr", int(s["Index"][0]))  # set speaker
        ZB.zBusTrigA(0,0,20) # Starts acquisition

        while  RX8[int(s["RX8"][0]) - 1].GetTagVal('playback'):
                time.sleep(0.05)
        RX8[int(s["RX8"][0]) - 1].SetTagVal("ch_nr",25)

        button, reaction_time = read_responsebox(RP2)
        response[count] = [int(i), button, reaction_time]
        count += 1
        print(count)
        time.sleep(1)

    for processor in RX8:
        processor.Halt()
    RP2.Halt()

    np.savetxt(os.environ["EXPDIR"]+os.environ["SUBJECT"]+"/results/freefield.txt", response)

    return response

def headphones(adapter=False):
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    RX8 = tdt.initialize_processor(processor="RX8", connection="GB", index=2, path=os.environ["EXPDIR"]+"rpvdsx/LED.rcx")
    RP2 = tdt.initialize_processor(processor="RP2", connection="GB", index=1, path=os.environ["EXPDIR"]+"rpvdsx/play_stereo.rcx")
    ZB = tdt.initialize_zbus("GB")
    if adapter:
        seq = np.loadtxt(os.environ["EXPDIR"]+os.environ["SUBJECT"]+"/lists/headphone_test_adapter.txt")
    else:
        seq = np.loadtxt(os.environ["EXPDIR"]+os.environ["SUBJECT"]+"/lists/headphone_test.txt")

    count=0
    response = np.zeros([len(seq),3]) #output array with three columns: stimulus, response and time
    for i in seq:
        left, right = prepare_stimuli(i, adapter)
        if count == 0:
            RP2.SetTagVal("playbuflen", len(left))
        RP2._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), 'left', 0, left) # = RP2.WriteTagV("left", 0, left)
        RP2._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)), 'right', 0, right)# = RP2.WriteTagV("right", 0, right)


        time.sleep(0.5)

        ZB.zBusTrigA(0, 0, 20)
        while RP2.GetTagVal("playback"):
            time.sleep(0.01)
        button, reaction_time = read_responsebox(RP2)

        #give visual Feedback if the response was wrong
        if button != int(i):
            led_nr = cfg["LEDs"][str(int(i))]
            RX8.SoftTrg(led_nr)
            print("ERROR FOR "+str(i))

        response[count]=[i, button, reaction_time]
        count+=1
        print(count)
    if adapter:
        np.savetxt(os.environ["EXPDIR"]+os.environ["SUBJECT"]+"/results/headphones_adapter.txt", response)
    else:
        np.savetxt(os.environ["EXPDIR"]+os.environ["SUBJECT"]+"/results/headphones.txt", response)


    RP2.Halt()
    RX8.Halt()

    return response


def read_responsebox(processor):
    import random
    bitval = 31  # bitmask value for no button pushed
    tic = time.time()
    tmp = [31,21,15,23,27]
    while bitval == 31:
        bitval = int(processor.GetTagVal("Response"))
        bitval = random.choice(tmp)
    if bitval == 30:
        button = 21
    elif bitval == 15:
        button = 23
    elif bitval == 23:
        button = 25
    elif bitval == 27:
        button = 27
    else:
        button = 0
    reaction_time = time.time() - tic

    return button, reaction_time

