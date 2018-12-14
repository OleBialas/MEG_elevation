import TDTblackbox as tdt
import time
from freefield_table import *
import json
from scipy.io import wavfile
import os
import numpy as np
from utilities import spectrum

def recording():
    cfg = json.load(open(os.environ["EXPDIR"] + "config.json"))
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

    adapter_left = make_adapter(left)
    adapter_right = make_adapter(right)

    write(left,right,adapter_left,adapter_right)


    RP2.Halt()
    for processor in RX8:
        processor.Halt()

    return


def make_adapter(sounds):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    cfg = json.load(open(os.environ["EXPDIR"] + "config.json"))
    n_samples = int(cfg["dur_adapter"]*cfg["FS"])+200
    # average spectrum over all speakers
    count = 0
    for sound in sounds:
        Z, phase, _ = spectrum(sound[0:n_samples], int(cfg["FS"]), log_power=True)
        if count == 0:
            Z_adapter = Z
        else:
            Z_adapter += Z
    Z_adapter /= len(cfg["speakers"])
    Z_adapter_smooth = lowess(Z_adapter, np.linspace(0,1,len(Z_adapter)), 0.02, return_sorted=False)
    Z_adapter_smooth = np.concatenate((Z_adapter_smooth, Z_adapter_smooth[::-1]))
    phase = 1j * np.random.randint(0, 6, len(Z_adapter_smooth)) + np.random.randn(len(Z_adapter_smooth))
    fft = Z_adapter_smooth*np.exp(1j*phase)
    adapter = np.real(np.fft.ifft(fft))[100:-100]
    return adapter

def write(left, right, adapter_left, adapter_right):
    # normalize adapter and recodrings while preserving IID
    cfg = json.load(open(os.environ["EXPDIR"] + "config.json"))
    rms = np.array([0,0], dtype=float) # rms for left and right
    #calculate difference in average RMS between left and right
    for l, r, i in zip(left,right,cfg["speakers"]):
        wavfile.write(os.environ["EXPDIR"] + os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_left_raw.wav",
                      int(cfg["FS"]), l)
        wavfile.write(os.environ["EXPDIR"] + os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_right_raw.wav",
                      int(cfg["FS"]),r)
        rms[0] += np.sqrt(np.mean(np.square(l)))
        rms[1] += np.sqrt(np.mean(np.square(r)))
    rms /= len(left)
    iid_factor = rms[1] / rms[0] # factor by which the recordings on the right are louder
    # normalize each recording (including adapter) with its own rms. Multiply recordings on the right with the factor to preserve IIDs
    count = 0
    for i in cfg["speakers"]:
        left[count] = left[count] / np.sqrt(np.mean(np.square(left[count])))
        right[count] = right[count] / np.sqrt(np.mean(np.square(right[count]))) * iid_factor
        wavfile.write(os.environ["EXPDIR"] + os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_left.wav",
                      int(cfg["FS"]), left[count])
        wavfile.write(os.environ["EXPDIR"] + os.environ["SUBJECT"] + "/recordings/speaker_" + str(i) + "_right.wav",
                      int(cfg["FS"]), right[count])
    adapter_left = adapter_left / np.sqrt(np.mean(np.square(adapter_left)))
    adapter_right = adapter_right / np.sqrt(np.mean(np.square(adapter_right)))* iid_factor
    wavfile.write(os.environ["EXPDIR"] + os.environ["SUBJECT"] + "/recordings/adapter_left.wav", int(cfg["FS"]),
                  adapter_left)
    wavfile.write(os.environ["EXPDIR"] + os.environ["SUBJECT"] + "/recordings/adapter_right.wav", int(cfg["FS"]),
                  adapter_right)