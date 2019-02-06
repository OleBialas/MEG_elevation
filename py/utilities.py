import json
import os
from scipy.io import wavfile
import numpy as np

def prepare_stimuli(speaker_nr, dur="long", adapter=False):

    cfg = json.load(open(os.environ["EXPDIR"] + "cfg/elevation.cfg"))
    left = wavfile.read(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+str(int(speaker_nr))+"_left.wav")[1]
    right=wavfile.read(os.environ["EXPDIR"]+"/data/"+os.environ["SUBJECT"]+"/recordings/speaker_"+str(int(speaker_nr))+"_right.wav")[1]
    if dur == "long":
        n_stimulus = int(cfg["dur_stimulus_long"]*cfg["FS"])
    if dur == "short":
        n_stimulus = int(cfg["dur_stimulus_short"]*cfg["FS"])

    if adapter == True:
        adapter_left = ramp(wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"] + "/recordings/adapter_left.wav")[1])
        adapter_right = ramp(wavfile.read(os.environ["EXPDIR"]+"data/"+os.environ["SUBJECT"] + "/recordings/adapter_right.wav")[1])
        n_adapter = len(adapter_left)

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

    return Z, freqs, phase # for some reason the last samples is really high and should be removed

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
        return

    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
        return

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y
