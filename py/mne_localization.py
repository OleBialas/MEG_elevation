from mne.beamformer import make_lcmv, apply_lcmv
from os import environ
from mne import compute_covariance, read_forward_solution, read_cov, convert_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse
from matplotlib import pyplot as plt
expdir = "C:/Projects/Elevation/bennewitz/"
environ["SUBJECTS_DIR"] = expdir+"/freesurfer/"


def beamformer(subject, evoked, fwd=None):
    """
    compute beamformer on evoked data
    :param subject: (str) subject name
    :param evoked: (mne.evoked.EvokedArray) evoked potential
    :return: source time course and lcmv fileter
    """
    if not fwd:
        fwd = read_forward_solution(expdir + subject + "/" + subject + "-fwd.fif")
    noise_cov = read_cov(expdir + subject + "/" + subject + "_noise-cov.fif")
    data_cov = read_cov(expdir + subject + "/" + subject + "_data-cov.fif")

    filter = make_lcmv(evoked.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                       pick_ori=None, weight_norm='unit-noise-gain')

    stc = apply_lcmv(evoked, filter, max_ori_out='signed')
    return stc, filter

def inverse_solution(subject, evoked, inv, snr=3.0, method="MNE", plot=True):

    lambda2 = 1.0 / snr ** 2
    stc = apply_inverse(evoked, inv, lambda2, method=method, verbose=True)

    if plot:
        plt.plot(1e3*stc.times, stc.data[::100,:].T)
        plt.xlabel("time (ms)")
        plt.ylabel("%s value" % method)
        plt.show()
        # plot cortex with point of maximum activation
        for hemi in ["rh","lh"]:
            vertno_max, time_max = stc.get_peak(hemi=hemi)
            brain=stc.plot(subject[:-1], hemi=hemi)
            brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color="blue", scale_factor=0.6, alpha=0.5)
    return stc


def inverse_operator(info, fwd, noise_cov):
    from mne.minimum_norm import make_inverse_operator
    inv = make_inverse_operator(info, fwd, noise_cov, loose=1, depth=None, verbose=True)
    return inv

def forward_operator(info, subject, write_fwd=False, mindist=2):
    """
    Compute forward solution for one subject with one head position (has to be computed separately for each block).
    trans file and surface model are loaded from the subject folder, so they need to be generated in advance.
    """
    from mne import read_trans, setup_source_space, read_bem_surfaces, make_bem_solution, make_forward_solution, write_forward_solution
    trans = read_trans(expdir + subject + "/" + subject + "2_raw-trans.fif")
    src = setup_source_space(subject=subject[:-1], spacing='oct6', add_dist=False)
    surfs = read_bem_surfaces(expdir + "freesurfer/" + subject[:-1] + "/bem/" + subject[:-1] + "-5120-bem.fif")
    bem = make_bem_solution(surfs)
    fwd = make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=mindist, n_jobs=2)
    if write_fwd:
        write_forward_solution(expdir + subject + "/" + subject + "-fwd.fif", fwd, overwrite=True)
    return fwd

def covariance(epochs, interval, plot=True):
    """
    compute (and plot) covariance matrix for epoched data in the given time window.
    """
    from mne.viz import plot_cov
    cov = compute_covariance(epochs, tmin=interval[0], tmax=interval[1], method="shrunk")
    if plot:
        plot_cov(cov, epochs.info)
    return cov

if __name__ == "__main__":
    from mne_erp import get_epochs, get_evokeds
    import json
    subject="el01b"
    blocks = [1,2,3]
    cfg = json.load(open("C:/Projects/Elevation/elevation.cfg"))
    source_estimates = []
    for i in blocks:
        epochs = get_epochs(subject,i,cfg["event_id"],cfg["epoch_interval"],cfg["baseline_interval"])
        fwd = forward_operator(epochs.info, subject=subject)
        noise_cov = covariance(epochs, cfg["baseline_interval"], plot=False)
        inv = inverse_operator(epochs.info, fwd, noise_cov)
        evokeds = get_evokeds(epochs, cfg["event_id"])
        count = 0
        for evoked in evokeds:
            stc = inverse_solution(subject, evoked,inv)
            if i == 1:
                source_estimates.append(stc)
            elif i > 1:
                source_estimates[count].data += stc.data


