import tdt
import time
from freefield_table import *
import json
from scipy.io import wavfile
import os
import numpy as np

def recording():
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    RX8=[]
    for i in [1,2]:
        RX8.append(tdt.initialize_processor(processor="RX8", connection="GB", index=i, path=os.environ["EXPDIR"] + "rpvdsx/play_noise.rcx"))
    RP2 = tdt.initialize_processor(processor="RP2", connection="GB", index=1, path=os.environ["EXPDIR"] + "rpvdsx/record_stereo.rcx")
    ZB = tdt.initialize_zbus(connection="GB")

    n_samples = int(cfg["dur_record"]*cfg["FS"])
    RP2.SetTagVal('recbuflen', n_samples)
    for processor in RX8:
        processor.SetTagVal("n_samples", n_samples)

    left = []
    right = []
    for i in cfg["speakers"]:

        s = freefield_table(Ongoing=[i])
        RX8[int(s["RX8"][0])-1].SetTagVal("ch_nr", int(s["Index"][0]))  # set speaker
        print("current speaker: " + str(i))
        ZB.zBusTrigA(0, 0, 20)  # Starts acquisition
        while RP2.GetTagVal('Recording'):
            time.sleep(0.05)
        RX8[int(s["RX8"][0]) - 1].SetTagVal("ch_nr", 25)  # set channel back to non-existent nr. 25
        # Read RP2 buffers and subtract the samples for the travel delay
        left.append(np.asarray(RP2.ReadTagV('SigInLeft', 0, n_samples))[1000:-1000]) #remove first and last 1000 samples
        right.append(np.asarray(RP2.ReadTagV('SigInRight', 0, n_samples))[1000:-1000])#to avoid on and offset effects

    for l, r, i in zip(left,right,cfg["speakers"]):
        wavfile.write(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_left_raw.wav",
                      int(cfg["FS"]), l)
        wavfile.write(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_right_raw.wav",
                      int(cfg["FS"]),r)

    RP2.Halt()
    for processor in RX8:
        processor.Halt()

    return


def normalize_recordings():
    "RMS normalization while preserving interaural intensity difference"
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    iid_factor = mean_iid()
    for i in cfg["speakers"]:
        l=wavfile.read(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_left_raw.wav")[1]
        r=wavfile.read(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_right_raw.wav")[1]
        wavfile.write((os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_right_norm.wav"),
        48828, r/np.sqrt(np.mean(np.square(r))))
        wavfile.write((os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_left_norm.wav"),
        48828, l/np.sqrt(np.mean(np.square(l)))/iid_factor)


def mean_iid():
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    rms = np.array([0,0], dtype=float) # rms for left and right
    for i in cfg["speakers"]:
        l=wavfile.read(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_left_raw.wav")[1]
        r=wavfile.read(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_right_raw.wav")[1]
        rms[0] += np.sqrt(np.mean(np.square(l)))
        rms[1] += np.sqrt(np.mean(np.square(r)))
    rms /= len(cfg["speakers"])
    iid_factor = rms[1] / rms[0]

    return iid_factor

def make_adapter():
    from statsmodels.nonparametric.smoothers_lowess import lowess
    cfg = json.load(open(os.environ["EXPDIR"] + "/cfg/elevation.cfg"))
    iid_factor = mean_iid()
    n_samples = int(cfg["dur_adapter"]*cfg["FS"])+200
    for side in ["left","right"]:
        adapter = np.zeros(n_samples)
        for i in cfg["speakers"]:
            sound=wavfile.read(os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_left_norm.wav")[1][0:n_samples]
            mag = np.abs(np.fft.fft(sound)) # magnitude of spectrum
            adapter += lowess(mag, np.linspace(0,1,len(mag)), 0.02, return_sorted=False) #smoothen
        adapter/=len(cfg["speakers"])
        phase = 1j * np.random.randint(0, 6, len(adapter)) + np.random.randn(len(adapter))
        adapter = np.real(np.fft.ifft(adapter*np.exp(1j*phase)))[100:-100]
        # normalize and save
        adapter /= np.sqrt(np.mean(np.square(adapter)))
        if side == "left":
            adapter /= iid_factor
        wavfile.write((os.environ["EXPDIR"] +"data/"+ os.environ["SUBJECT"] + "/recordings/adapter_"+side+".wav"),
        48828, adapter)
