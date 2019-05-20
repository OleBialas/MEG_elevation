import os
import numpy as np
from matplotlib import pyplot as plt
from mne.io import read_raw_fif, RawArray
from mne.io.constants import FIFF
from mne import read_trans, read_bem_surfaces, create_info
from mayavi import mlab
from tkinter import *

class align_electrodes():

    def __init__(self, capsize=54, subject=None):
        if not subject:
            subject=os.environ["SUBJECT"]

        cap = read_raw_fif(os.environ["RAWDIR"]+"acticap_64_ch_size_"+str(capsize)+".fif")
        info = read_raw_fif(os.path.join(os.environ["RAWDIR"],subject, subject+"1s.fif")).info #kind=2 means nasion
        self.nasion = list(filter(lambda dig: dig["kind"] ==1 and dig["ident"]==2, info["dig"] ))[0]["r"]
        self.electrodes = np.array([ch["loc"][0:3] for ch in cap.info["chs"]]) # first 3 = channel pos, next three=eference
        self.reference = cap.info["chs"][0]["loc"][3:6]
        trans = read_trans(os.path.join(os.environ["EXPDIR"],subject,subject+"1l_raw-trans.fif"))["trans"]
        self.surface = read_bem_surfaces(os.path.join(os.environ["SUBJECTS_DIR"], subject[0:-1], "bem", subject[0:-1] + "-5120-bem.fif"))[0]
        ones =np.expand_dims(np.ones(len(self.surface["rr"])),axis=1)
        # add a column of ones to bem, then apply tranformation, then throw away las 3 columns
        self.surface["rr"] = np.matmul(np.hstack((self.surface["rr"],ones)),trans)[:,0:3]
        self.plot_alignment()
        # start the gui. 3 Buttons for rotating, 3 buttons for scaling
        root=Tk()

        alpha_x = DoubleVar()
        label_rx = Label(root, text="rotate around x by:").grid(row=0,column=0)
        entry_rx = Entry(root, textvariable=alpha_x).grid(row=0, column=1)
        button_rx = Button(root, text="go", command=lambda:self.rotate_x(alpha_x.get())).grid(row=0, column=2)

        alpha_y = DoubleVar()
        label_ry = Label(root, text="rotate around y by:").grid(row=1,column=0)
        entry_ry = Entry(root, textvariable=alpha_y).grid(row=1, column=1)
        button_ry = Button(root, text="go", command=lambda:self.rotate_y(alpha_y.get())).grid(row=1, column=2)

        alpha_z = DoubleVar()
        label_rz = Label(root, text="rotate around z by:").grid(row=2,column=0)
        entry_rz = Entry(root, textvariable=alpha_z).grid(row=2, column=1)
        button_rz = Button(root, text="go", command=lambda:self.rotate_z(alpha_z.get())).grid(row=2, column=2)

        scaler_x = DoubleVar()
        label_sx = Label(root, text="scale x by:").grid(row=3,column=0)
        entry_sx = Entry(root, textvariable=scaler_x).grid(row=3, column=1)
        button_sx = Button(root, text="go", command=lambda:self.scale_x(scaler_x.get())).grid(row=3, column=2)

        scaler_y = DoubleVar()
        label_sy = Label(root, text="scale y by:").grid(row=4,column=0)
        entry_sy = Entry(root, textvariable=scaler_y).grid(row=4, column=1)
        button_sy = Button(root, text="go", command=lambda:self.scale_y(scaler_y.get())).grid(row=4, column=2)

        scaler_z = DoubleVar()
        label_sz = Label(root, text="scale z by:").grid(row=5,column=0)
        entry_sz = Entry(root, textvariable=scaler_z).grid(row=5, column=1)
        button_sz = Button(root, text="go", command=lambda:self.scale_z(scaler_z.get())).grid(row=5, column=2)

        root.mainloop() # start the gui

    def rotate_x(self, alpha):
        Rx = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)] ])
        self.electrodes = np.matmul(self.electrodes, Rx)
        self.reference = np.matmul(self.reference, Rx)
        self.update_plot()

    def rotate_y(self, alpha):
        Ry = np.array([[np.cos(alpha), 0, np.sin(alpha)],[0, 1, 0],[-np.sin(alpha), 0, np.cos(alpha)]])
        self.electrodes = np.matmul(self.electrodes, Ry)
        self.reference = np.matmul(self.reference, Ry)
        self.update_plot()

    def rotate_z(self, alpha):
        Rz = np.array([[np.cos(alpha), np.sin(alpha), 0],[-np.sin(alpha), np.cos(alpha), 0],[0, 0, 1]])
        self.electrodes = np.matmul(self.electrodes, Rz)
        self.reference = np.matmul(self.reference, Rz)
        self.update_plot()

    def scale_x(self, fx):
        Sx = np.array([[fx, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.electrodes = np.matmul(self.electrodes, Sx)
        self.reference = np.matmul(self.reference, Sx)
        self.update_plot()

    def scale_y(self, fy):
        Sy = np.array([[1, 0, 0], [0, fy, 0], [0, 0, 1]])
        self.electrodes = np.matmul(self.electrodes, Sy)
        self.reference = np.matmul(self.reference, Sy)
        self.update_plot()

    def scale_z(self, fz):
        Sz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, fz]])
        self.electrodes = np.matmul(self.electrodes, Sz)
        self.reference = np.matmul(self.reference, Sz)
        self.update_plot()

    def plot_alignment(self):

        self.fig = mlab.figure()
        x, y, z = self.surface['rr'].T
        mesh = mlab.pipeline.triangular_mesh_source(x, y, z, self.surface['tris'], scalars=None, figure=self.fig)
        mesh = mlab.pipeline.poly_data_normals(mesh)
        mesh.filter.compute_cell_normals = False
        mesh.filter.consistency = False
        mesh.filter.non_manifold_traversal = False
        mesh.filter.splitting = False
        surface = mlab.pipeline.surface(mesh, color=(1,1,1),figure=self.fig) # Plot bem surface --> see mne.viz._3d._create_mesh_surf
        self.electrode_points = mlab.points3d(self.electrodes[:, 0], self.electrodes[:, 1], self.electrodes[:, 2],scale_factor=0.01, figure=self.fig) # plot points
        self.reference_point= mlab.points3d(self.reference[0], self.reference[1], self.reference[2],scale_factor=0.01, figure=self.fig, color=(0,1,0))
        self.nasion_point = mlab.points3d(self.nasion[0], self.nasion[1], self.nasion[2],scale_factor=0.01, figure=self.fig, color=(0,0,1)) # plot points

    def update_plot(self):

        self.electrode_points.remove()
        self.reference_point.remove()
        self.electrode_points = mlab.points3d(self.electrodes[:, 0], self.electrodes[:, 1], self.electrodes[:, 2],scale_factor=0.01, figure=self.fig) # plot points
        self.reference_point= mlab.points3d(self.reference[0], self.reference[1], self.reference[2],scale_factor=0.01, figure=self.fig, color=(0,1,0))

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.environ["PYDIR"])
    from scale_electrodes import *
    a = align_electrodes(subject="el04a")
    a.plot_alignment()
