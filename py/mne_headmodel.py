import os
import json
import sys
sys.path.append(os.environ["PYDIR"])
from mne import read_bem_surfaces, read_trans, setup_source_space, make_forward_solution, make_bem_solution, \
     write_forward_solution, compute_covariance, write_cov
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from mne.io import read_raw_fif
import numpy as np
from copy import deepcopy
from mne_preprocessing import load_epochs




def inverse_operator(epochs, tnoise=(0.9,1), part="b", write_inv=True, write_fwd=True, write_cov=True, mindist=2):

    noise_cov = compute_covariance(epochs, tmin=tnoise[0], tmax=tnoise[1], method="shrunk")
    if write_cov:
        write_cov(expdir + subject+part + "/" + subject+part + "-fwd.fif", noise_cov)
    inv = make_inverse_operator(epochs.info, fwd, noise_cov, loose=1, depth=None, verbose=True)
    if write_inv:
        write_inverse_operator(expdir + subject+part + "/" + subject+part + "-inv.fif", inv)
    return inv


def make_fwd(write=True, mindist=2):

    info = read_raw_fif(os.path.join(os.environ["EXPDIR"],os.environ["RAWDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+str(block)+".fif")).info
    trans=get_trans()
    subject = os.environ["SUBJECT"][:-1]
    src = setup_source_space(subject=subject, spacing='oct6', add_dist=False)
    surfs = read_bem_surfaces(os.path.join(os.environ["SUBJECTS_DIR"], subject, "bem", subject + "-5120-bem.fif"))
    bem = make_bem_solution(surfs)
    fwd = make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=mindist, n_jobs=2)
    if write:
        write_forward_solution(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+".fwd"),fwd)
    return fwd


def get_trans():
    files = os.listdir(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"]))
    for file in files:
        if "trans" in file:
            trans = read_trans(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],file))
    return trans


def scale_forward_model(info, subject):

    """
    Compute a forward model by scaling the bem from the scans of subject "col27" to the fiducial points of the subject
    given as input
    :param info: (mne.io.meas_info.Info) info from the subjects raw data
    :param subject: (str) name of the subject
    :return: forward model
    """
    reference = transform(np.array([[-84.2, -1, -67], [0, 101.2, -60], [82.1, -1, -67]]) / 1000, shift=True)
    # load the ingredients for the model
    cardinals_subject = np.array(
        [info["dig"][0]["r"], info["dig"][1]["r"], info["dig"][2]["r"]])
    trans = read_trans(expdir + subject + "/" + subject + "1_raw-trans.fif")
    trans["trans"] = np.identity(4)  # don't do any automatic transformation
    src = setup_source_space(subject="col27", spacing='oct6', add_dist=False)
    surfs = read_bem_surfaces(expdir + "freesurfer/col27/bem/col27-5120-bem.fif")

    # convert source space and bem from the MRI to the fiducial system ( points ["nn"] need to be shifted and rotated,
    # direction vectors ["rr"] only need to be rotated) and scale them:

    for surface in surfs:
        surface["rr"] = transform(surface["rr"], shift=True)
        surface["nn"] = transform(surface["nn"], shift=False)
        surface["rr"] = scale(surface["rr"], cardinals_subject, reference)
    bem = make_bem_solution(surfs)

    # model source space as a shrunk version of the bem surface:
    # source space needs to have two hemispheres so one is set to 0
    cortical_surface = deepcopy(surfs[0])
    cortical_surface["rr"] *= 0.9

    src[0]["inuse"] = np.ones(cortical_surface["np"], dtype=np.int8)
    src[0]["nn"] = cortical_surface["nn"]
    src[0]["np"] = cortical_surface["np"]
    src[0]["ntri"] = cortical_surface["ntri"]
    src[0]["nuse"] = cortical_surface["np"]
    src[0]["nuse_tri"] = cortical_surface["ntri"]
    src[0]["rr"] = cortical_surface["rr"]
    src[0]["tris"] = cortical_surface["tris"]
    src[0]["use_tris"] = cortical_surface["tris"]
    src[0]["vertno"] = range(cortical_surface["np"])

    src[1]["inuse"] = np.zeros(cortical_surface["np"], dtype=np.int8)
    src[1]["nn"] = cortical_surface["nn"]
    src[1]["np"] = cortical_surface["np"]
    src[1]["ntri"] = cortical_surface["ntri"]
    src[1]["nuse"] = 0
    src[1]["nuse_tri"] = 0
    src[1]["rr"] = cortical_surface["rr"]
    src[1]["tris"] = cortical_surface["tris"]
    src[1]["use_tris"] = np.expand_dims(np.array([]), axis=1)
    src[1]["vertno"] = np.array([])

    fwd = make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=2, n_jobs=2)
    write_forward_solution(expdir + subject + "/" + subject + "-fwd.fif", fwd, overwrite=True)

    return fwd
def get_sensors(info):
    """
    :param info: mne.raw.info structure
    :return: array of x,y,z coordinates of all sensors

    get array of points describing the MEG sensors
    sensors are described as one vector that is shifted (by a vector S) to different postions and rotated
    (by a matrix R).
    The resulting vector must then be transformed (by a matrix T) from the device to the head coordinate system:

    R R R S           0             T T T T           x
    R R R S     x     0     x       T T T T     =     x
    R R R S           0             T T T T           x
    0 0 0 1           1             T T T T           x

    The first three elements of the resulting vector x are the coordinates of the sensor
    """
    T = list(info["dev_head_t"].values())[2]
    all_channels = deepcopy(info["chs"])
    # get sensor coordinates (shift vector + rotation matrix)
    ch_locs = np.zeros([len(info["chs"]), 3])
    count = 0
    for channel in all_channels:
        channel["loc"] = np.reshape(channel["loc"], [3, 4], order="F")  # reshape into 3 by 4 matrix
        # now move the left column (shift vector) to the right
        # by adding it to the right end then removing it from the left end:
        shift = np.expand_dims(channel["loc"][:, 0], 1)
        channel["loc"] = np.append(channel["loc"], shift, axis=1)
        channel["loc"] = np.delete(channel["loc"], 0, 1)
        channel["loc"] = np.vstack([channel["loc"], [0, 0, 0, 1]])
        channel["loc"] = np.matmul(np.matmul(T, channel["loc"]), [0, 0, 0, 1])
        ch_locs[count] = channel["loc"][0:3]
        count += 1
    return ch_locs


def transform(points, shift=True):
    """
    input: --> canoncial MRI coordinates
    point: array of points with x,y and z coordinates
    reference: x,y,z coordinates of ears and nose of the head used to compute the origin and rotate the points
    return : --> fiducial coordinates
    shifted and rotated coordinates"""
    reference = np.array([[-84.2, -1, -67], [0, 101.2, -60], [82.1, -1, -67]])/1000

    # calculate origin as O = LE + (LE - RE) [(N - LE) · (LE - RE)] / ||LE - RE||²
    O = reference[0] + (reference[0] - reference[2]) * np.dot(
        (reference[1] - reference[0]), (reference[0] - reference[2])) \
        / np.dot((reference[0] - reference[2]),
                 (reference[0] - reference[2]))

    # define unity vectors for new axes + rotation matrix:
    ey = (reference[1] - O) / np.linalg.norm(reference[1] - O)  # ex = (N - O) / ||N - O||
    ex = -(reference[0] - O) / np.linalg.norm(reference[0] - O)  # ey = (LE - O) / ||LE - O||
    ez = np.cross(ex, ey)  # ez = ex x ey
    R = np.transpose([ex, ey, ez])  # rotation matrix
    # now every point can be transformed from the canonical MRI to feducial system:
    # P_fiducial = R * (P_canonical - O)

    if shift == True:
        points = np.matmul((points - O), R)
    else:
        points = np.matmul(points, R)

    return points


def scale(surface, cardinals_subject,reference):
    """Scale BEM surface to the head of the subject. Surface must be converted to the fiducial system before
    input:
    surface: template BEM surface to be scaled
    cardinals_subject: x,y,z coordinates of ears and nose from subject
    cardinals_template: x,y,z coordinates of ears and nose from template
    """
    #reference = np.array([[-84.2, -1, -67], [0, 101.2, -60], [82.1, -1, -67]])

    # scaling matrix S:
    sx = np.linalg.norm(cardinals_subject[2, :] - cardinals_subject[0, :]) / np.linalg.norm(
        reference[2, :] - reference[0, :])
    sy = np.linalg.norm(cardinals_subject[1, :]) / np.linalg.norm(reference[1, :])
    sz = (sx + sy) / 2
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])  # scaling matrix
    print("scaling factors for each axis are: \n x:"+str(sx)+" \n y: "+str(sy)+"  \n z: "+str(sz))
    # add an extra dimension with zeros to the bem surface then scale
    surface = np.matmul(surface, S)
    return surface

if __name__ == "__main__":
    from mne.io import read_raw_fif
    from mne.epochs import Epochs
    from mne import read_events
    subject="el01"
    raw = read_raw_fif("C:/Projects/Elevation/bennewitz/el01b/el01b1.fif", preload=True)
    events = read_events("C:/Projects/Elevation/bennewitz/el01b/el01b1_cor.eve")
    epochs = Epochs(raw, events, tmin=-0.1, tmax=1.0, baseline=(-0.1,0), preload=True)
    inverse_operator(epochs,subject)
