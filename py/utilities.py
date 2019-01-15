import json
import os
from scipy.io import wavfile
import numpy as np

def prepare_stimuli(speaker_nr, dur="long", adapter=False):

    cfg = json.load(open(os.environ["EXPDIR"] + "cfg/elevation.cfg"))
<<<<<<< HEAD
    left = wavfile.read(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+str(int(speaker_nr))+"_left.wav")[1]
    right=wavfile.read(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+str(int(speaker_nr))+"_right.wav")[1]
    n = len(left)
    if dur == "long":
        n_stimulus = int(cfg["dur_stimulus_long"]*cfg["FS"])
    if dur == "short":
        n_stimulus = int(cfg["dur_stimulus_short"]*cfg["FS"])


=======
    left = wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+str(int(speaker_nr))+"_left.wav")[1]
    right=wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+str(int(speaker_nr))+"_right.wav")[1]

    n = len(left)
    if adapter == True:
        adapter_left = ramp(wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"] + "/recordings/adapter_left.wav")[1])
        adapter_right = ramp(wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"] + "/recordings/adapter_right.wav")[1])
        n_stimulus = int(cfg["dur_stimulus"]*cfg["FS"])
        n_adapter = len(adapter_left)
    else:
        n_stimulus = int(cfg["dur_freefield"] * cfg["FS"])
>>>>>>> 88acfde2916c33dfa4506c8c1c36c353b8f9d9cb
    # pick random segment from recorded stimulus
    start = np.random.randint(0, len(left)-n_stimulus)
    left = ramp(left[start:start+n_stimulus])
    right = ramp(right [start:start+n_stimulus])

    if adapter == True: # crossfade adapter and stimulus
        adapter_left = ramp(wavfile.read(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"] + "/recordings/adapter_left.wav")[1])
        adapter_right = ramp(wavfile.read(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"] + "/recordings/adapter_right.wav")[1])
        n_adapter = len(adapter_left)
        n_samples = n_stimulus+n_adapter - int(cfg["dur_ramp"] * cfg["FS"]) # minus duration of ramp because ramps should overlap
        left = np.concatenate((np.zeros(n_samples-len(left)),left))
        adapter_left = np.concatenate((adapter_left,np.zeros(n_samples-len(adapter_left))))
        left = adapter_left+left
        right = np.concatenate((np.zeros(n_samples-len(right)),right))
        adapter_right = np.concatenate((adapter_right,np.zeros(n_samples-len(adapter_right))))
        right = adapter_right+right

    return left, right

def ramp(sound):
    cfg = json.load(open(os.environ["EXPDIR"] + "cfg/elevation.cfg"))

    n_ramp = int(cfg["dur_ramp"]*cfg["FS"])
    if n_ramp %2 ==0:
        n_ramp+=1
    ramp = np.linspace(0,1,n_ramp)
    envelope = np.concatenate((ramp,np.ones(len(sound)-n_ramp*2),ramp[::-1]))
    sound = sound*envelope

    return sound

def spectrum(x, FS, log_power=False):

    n = len(x)
    if n%2 != 0:
        x = x[0:-1]
        n = len(x)
    fftx = np.fft.fft(x)  # take the fourier transform
    pxx = np.abs(fftx)  # only keep the magnitude
    nUniquePts = int(np.ceil((n + 1) / 2))
    pxx = pxx[0:nUniquePts]
    pxx = pxx / n  # scale by the number of points so that the magnitude does not depend on the length of the signal
    pxx = pxx ** 2  # square to get the power
    pxx[
    1:] *= 2  # we dropped half the FFT, so multiply by 2 to keep the same energy, except at the DC term at p[0] (which is unique) (not sure if necessary with rfft! CHECK!)
    freqs = np.linspace(0, 1, len(pxx)) * (FS / 2)
    phase = np.unwrap(np.mod(np.angle(fftx), 2 * np.pi))
    Z = pxx

    if log_power:
        Z[Z < 1e-20] = 1e-20  # no zeros because we take logs
        Z = 10 * np.log10(Z)

    return Z[0:-2], freqs[0:-2], phase[0:-2] # for some reason the last samples is really high and should be removed