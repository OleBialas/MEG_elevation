import os
import numpy as np
from matplotlib import pyplot as plt
from mne.io import read_raw_fif
from mne.io.constants import FIFF
from mne import read_trans, read_bem_surfaces
from mayavi import mlab
os.environ["SUBJECT"]="el04a"
info = read_raw_fif(os.path.join(os.environ["RAWDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"1s.fif")).info
#for testing purpose act as if digitization points were eeg electrodes:
eeg_loc = np.array([point["r"] for point in info["dig"]])


trans = read_trans(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"1l_raw-trans.fif"))["trans"]
bem = read_bem_surfaces(os.path.join(os.environ["SUBJECTS_DIR"], os.environ["SUBJECT"][0:-1], "bem", os.environ["SUBJECT"][0:-1] + "-5120-bem.fif"))[0]

#transform bem surface to head coordinate system:
ones =np.expand_dims(np.ones(len(bem["rr"])),axis=1)
points = np.hstack((bem["rr"],ones)) # add a column of ones
points = np.matmul(points,trans) # apply transformation
bem["rr"] = points[:,0:3] # throw the last column away
# plot bem and electrode points
fig = mlab.figure()
x, y, z = bem['rr'].T
mesh = mlab.pipeline.triangular_mesh_source(x, y, z, bem['tris'], scalars=None, figure=fig)
mesh = mlab.pipeline.poly_data_normals(mesh)
mesh.filter.compute_cell_normals = False
mesh.filter.consistency = False
mesh.filter.non_manifold_traversal = False
mesh.filter.splitting = False
surface = mlab.pipeline.surface(mesh, color=(1,1,1),figure=fig) # Plot bem surface --> see mne.viz._3d._create_mesh_surf
points = mlab.points3d(eeg_loc[:, 0], eeg_loc[:, 1], eeg_loc[:, 2],scale_factor=0.01, figure=fig) # plot points
