
from mne.transforms import rotation, rotation3d, scaling, translation
import numpy as np
from functools import reduce

def rotate_x(data, alpha):
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)] ])
    return Rx

def rotate_y(data, alpha):
    Ry = np.array([[np.cos(alpha), 0, np.sin(alpha)],[0, 1, 0],[-np.sin(alpha), 0, np.cos(alpha)]])
    return Rx

def rotate_z(data, alpha):
    Rz = np.array([[np.cos(alpha), np.sin(alpha), 0],[-np.sin(alpha), np.cos(alpha), 0],[0, 0, 1]])
    return Rx

def scale(data, fx, fy, fz):
    S=np.array([[fx, 0, 0], [0, fy, 0], [0, 0, fz]])


def pol2cart(r, phi):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return np.squeeze([x, y]).T

def cart2sph(xyz): # ISO convention
    #xyz = np.expand_dims(xyz, 0)
    tmp = xyz[:, 0]**2 + xyz[:, 1]**2
    r = np.sqrt(tmp + xyz[:, 2]**2)
    theta = np.arctan2(np.sqrt(tmp), xyz[:,2]) # for theta angle defined from Z-axis down
    # theta = np.arctan2(xyz[:,2], np.sqrt(xy)) # for theta angle defined from XY-plane up
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    return r, theta, phi

def sph2cart(r, theta, phi):
    return np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta, )*np.sin(phi), r*np.cos(theta)])

def normalize_rot(rot):

    q = rot_to_quat(rot)
    q /= np.sqrt(np.sum(q**2))
    return quat_to_rot(q)

def invert(trans):

    rot = normalize_rot(trans[:3,:3].flatten())
    result = np.zeros((4,4))
    result[3,3] = 1.0
    t = -rot.T.dot(trans[:3, 3])
    result[:3, :3] = rot.T
    result[:3, 3] = t
    return result

def quat_to_rot(q):
    if q.size == 3:
        q = np.hstack([np.sqrt(1 - np.sum(q**2)), q])
    rot = np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
                   [2*(q[2]*q[1] + q[0]*q[3]), q[0]**2 - q[1]**2 + q[2 ]**2 - q[3]**2, 2*(q[2]*q[3]- q[0]*q[1])],
                   [2*(q[3]*q[1] - q[0]*q[2]), 2*(q[3]*q[2] + q[0]*q[1 ]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]])
    return rot

def rot_to_quat(rot):
    """Convert a rotation matrix to quaternions"""
    # see e.g. http://www.euclideanspace.com/maths/geometry/rotations/
    #                 conversions/matrixToQuaternion/
    t = 1. + rot[0] + rot[4] + rot[8]
    if t > np.finfo(rot.dtype).eps:
        s = np.sqrt(t) * 2.
        qx = (rot[7] - rot[5]) / s
        qy = (rot[2] - rot[6]) / s
        qz = (rot[3] - rot[1]) / s
        qw = 0.25 * s
    elif rot[0] > rot[4] and rot[0] > rot[8]:
        s = np.sqrt(1. + rot[0] - rot[4] - rot[8]) * 2.
        qx = 0.25 * s
        qy = (rot[1] + rot[3]) / s
        qz = (rot[2] + rot[6]) / s
        qw = (rot[7] - rot[5]) / s
    elif rot[4] > rot[8]:
        s = np.sqrt(1. - rot[0] + rot[4] - rot[8]) * 2
        qx = (rot[1] + rot[3]) / s
        qy = 0.25 * s
        qz = (rot[5] + rot[7]) / s
        qw = (rot[2] - rot[6]) / s
    else:
        s = np.sqrt(1. - rot[0] - rot[4] + rot[8]) * 2.
        qx = (rot[2] + rot[6]) / s
        qy = (rot[5] + rot[7]) / s
        qz = 0.25 * s
        qw = (rot[3] - rot[1]) / s
    return np.array((qw, qx, qy, qz))

def registration(s, t):

    s_mean = np.mean(s, axis=0)
    t_mean = np.mean(t, axis=0)
    s_c = s - s_mean
    t_c = t - t_mean

    H = s_c.T.dot(t_c)
    P = np.array([[H[0,0] + H[1,1] + H[2,2], H[1,2] - H[2,1], H[2,0] - H[0,2], H[0,1] - H[1,0]],
                  [H[1,2] - H[2,1], H[0,0] - H[1,1] - H[2,2], H[0,1] + H[1,0], H[2,0] + H[0,2]],
                  [H[2,0] - H[0,2], H[0,1] + H[1,0], H[1,1] - H[0,0] - H[2,2], H[1,2] + H[2,1]],
                  [H[0,1] - H[1,0], H[2,0] + H[0,2], H[1,2] + H[2,1], H[2,2] - H[0,0] - H[1,1]]])
    v, W = np.linalg.eig(P)
    a = np.argmax(v)
    q = W[:, a]
    R = np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
                   [2*(q[2]*q[1] + q[0]*q[3]), q[0]**2 - q[1]**2 + q[2 ]**2 - q[3]**2, 2*(q[2]*q[3]- q[0]*q[1])],
                   [2*(q[3]*q[1] - q[0]*q[2]), 2*(q[3]*q[2] + q[0]*q[1 ]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]])

    res = t_c - s_c.dot(R.T) # Transposed notation because of the shape of the point matrices
    t = t_mean - R.dot(s_mean)
    if q[0] < 0.:
        q *= -1.
    return q, t

def point_cloud_error(source, target):
    from scipy.spatial.distance import cdist
    Y = cdist(source, target, 'euclidean')
    dist = Y.min(axis=1)
    return dist

def _point_cloud_error_balltree(src_pts, tgt_tree):
    """Find the distance from each source point to its closest target point.
    Uses sklearn.neighbors.BallTree for greater efficiency
    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Source points.
    tgt_tree : sklearn.neighbors.BallTree
        BallTree of the target points.
    Returns
    -------
    dist : array, shape = (n, )
        For each point in ``src_pts``, the distance to the closest point in
        ``tgt_pts``.
    """
    dist, _ = tgt_tree.query(src_pts)
    return dist.ravel()

def _trans_from_params(param_info, params):
    """Convert transformation parameters into a transformation matrix.
    Parameters
    ----------
    param_info : tuple,  len = 3
        Tuple describing the parameters in x (do_translate, do_rotate,
        do_scale).
    params : tuple
        The transformation parameters.
    Returns
    -------
    trans : array, shape = (4, 4)
        Transformation matrix.
    """
    do_rotate, do_translate, do_scale = param_info
    i = 0
    trans = []

    if do_rotate:
        x, y, z = params[:3]
        trans.append(rotation(x, y, z))
        i += 3

    if do_translate:
        x, y, z = params[i:i + 3]
        trans.insert(0, translation(x, y, z))
        i += 3

    if do_scale == 1:
        s = params[i]
        trans.append(scaling(s, s, s))
    elif do_scale == 3:
        x, y, z = params[i:i + 3]
        trans.append(scaling(x, y, z))

    trans = reduce(np.dot, trans)
    return trans

def fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=True,
                    scale=0, x0=None, leastsq_args={}, out='params'):
    """Find a transform between unmatched sets of points.
    This minimizes the squared distance from each source point to its closest
    target point, using :func:`scipy.optimize.leastsq` to find a
    transformation using rotation, translation, and scaling (in that order).
    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Points to which the transform should be applied.
    tgt_pts : array, shape = (m, 3)
        Points to which src_pts should be fitted. Each point in tgt_pts should
        correspond to the point in src_pts with the same index.
    rotate : bool
        Allow rotation of the ``src_pts``.
    translate : bool
        Allow translation of the ``src_pts``.
    scale : 0 | 1 | 3
        Number of scaling parameters. With 0, points are not scaled. With 1,
        points are scaled by the same factor along all axes. With 3, points are
        scaled by a separate factor along each axis.
    x0 : None | tuple
        Initial values for the fit parameters.
    leastsq_args : dict
        Additional parameters to submit to :func:`scipy.optimize.leastsq`.
    out : 'params' | 'trans'
        In what format to return the estimate: 'params' returns a tuple with
        the fit parameters; 'trans' returns a transformation matrix of shape
        (4, 4).
    Returns
    -------
    x : array, shape = (n_params, )
        Estimated parameters for the transformation.
    Notes
    -----
    Assumes that the target points form a dense enough point cloud so that
    the distance of each src_pt to the closest tgt_pt can be used as an
    estimate of the distance of src_pt to tgt_pts.
    """
    from scipy.optimize import leastsq
    kwargs = {'epsfcn': 0.01}
    kwargs.update(leastsq_args)

    # assert correct argument types
    src_pts = np.atleast_2d(src_pts)
    tgt_pts = np.atleast_2d(tgt_pts)
    translate = bool(translate)
    rotate = bool(rotate)
    scale = int(scale)

    if translate:
        src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))

    try:
        from sklearn.neighbors import BallTree
        tgt_pts = BallTree(tgt_pts)
        errfunc = _point_cloud_error_balltree
    except ImportError:
        warn("Sklearn could not be imported. Fitting points will be slower. "
             "To improve performance, install the sklearn module.")
        errfunc = _point_cloud_error

    # for efficiency, define parameter specific error function
    param_info = (rotate, translate, scale)
    if param_info == (True, False, 0):
        x0 = x0 or (0, 0, 0)

        def error(x):
            rx, ry, rz = x
            trans = rotation3d(rx, ry, rz)
            est = np.dot(src_pts, trans.T)
            err = errfunc(est, tgt_pts)
            return err
    elif param_info == (True, False, 1):
        x0 = x0 or (0, 0, 0, 1)

        def error(x):
            rx, ry, rz, s = x
            trans = rotation3d(rx, ry, rz) * s
            est = np.dot(src_pts, trans.T)
            err = errfunc(est, tgt_pts)
            return err
    elif param_info == (True, False, 3):
        x0 = x0 or (0, 0, 0, 1, 1, 1)

        def error(x):
            rx, ry, rz, sx, sy, sz = x
            trans = rotation3d(rx, ry, rz) * [sx, sy, sz]
            est = np.dot(src_pts, trans.T)
            err = errfunc(est, tgt_pts)
            return err
    elif param_info == (True, True, 0):
        x0 = x0 or (0, 0, 0, 0, 0, 0)

        def error(x):
            rx, ry, rz, tx, ty, tz = x
            trans = np.dot(translation(tx, ty, tz), rotation(rx, ry, rz))
            est = np.dot(src_pts, trans.T)
            err = errfunc(est[:, :3], tgt_pts)
            return err
    else:
        raise NotImplementedError(
            "The specified parameter combination is not implemented: "
            "rotate=%r, translate=%r, scale=%r" % param_info)

    est, _, info, msg, _ = leastsq(error, x0, full_output=True, **kwargs)

    if out == 'params':
        return est
    elif out == 'trans':
        return _trans_from_params(param_info, est)
    else:
        raise ValueError("Invalid out parameter: %r. Needs to be 'params' or "
                         "'trans'." % out)
