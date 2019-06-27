import os
import numpy as np
from matplotlib import pyplot as plt
from mne.io import read_raw_fif, RawArray
from mne.io.constants import FIFF
from mne import read_trans, read_bem_surfaces, create_info, read_epochs
from mayavi import mlab
from tkinter import *
from scipy.optimize import leastsq
from sklearn.neighbors import BallTree
import sys
sys.path.append(os.environ["PYDIR"])
from transform import *
from mne.channels import DigMontage, Montage

class align_electrodes():

    def __init__(self, capsize=56, subject=None):

        if not subject:
            self.subject=os.environ["SUBJECT"]
        else:
            os.environ["SUBJECT"]=subject
        self.capsize=capsize
        self.nasion=np.array([0,0,0], dtype=float)
        self.load_electrodes()
        self.surface = read_bem_surfaces(os.path.join(os.environ["SUBJECTS_DIR"], subject, "bem", subject + "-head.fif"))[0]
        ones =np.expand_dims(np.ones(len(self.surface["rr"])),axis=1)
        self.tgt_tree = BallTree(self.surface["rr"])

    def align_nasion(self):
        self.points = self.nasion
        self.pts_type = "nasion"
        self.start_gui()
        self.nasion = self.points

    def align_electrodes(self):
        self.points = self.electrodes
        self.pts_type = "electrodes"
        self.start_gui()
        self.electrodes = self.points

    def load_electrodes(self):

        cap = read_raw_fif(os.environ["RAWDIR"]+"acticap_64_ch_size_"+str(self.capsize)+".fif")
        self.electrodes = np.array([ch["loc"][0:3] for ch in cap.info["chs"]]) # first 3 = channel pos, next three=eference
        self.ch_names = cap.info["ch_names"]
        ground =  np.array(cap.info["dig"][0]["r"], ndmin=2)
        reference = np.array(cap.info["chs"][0]["loc"][3:6], ndmin=2)
        #last entry is ground, second last is reference
        self.electrodes = np.append(self.electrodes,np.array([reference[0],ground[0]]),axis=0)

    def start_gui(self):

        self.plot_alignment()
        root=Tk() # start the gui. 3 Buttons for rotating, 3 buttons for scaling
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

        sx, sy, sz = DoubleVar(),DoubleVar(), DoubleVar()
        label_scale = Label(root, text="scale x, y and z by a factor of:").grid(row=3,column=1)
        entry_sx = Entry(root, textvariable=sx).grid(row=4, column=0)
        entry_sy = Entry(root, textvariable=sy).grid(row=4, column=1)
        entry_sz = Entry(root, textvariable=sz).grid(row=4, column=2)
        button_scale = Button(root, text="scale", command=lambda:self.scale(sx.get(),sy.get(),sz.get())).grid(row=4, column=3)


        shift_x = DoubleVar()
        label_shx = Label(root, text="shift x by:").grid(row=6,column=0)
        entry_shx = Entry(root, textvariable=shift_x).grid(row=7, column=0)

        shift_y = DoubleVar()
        label_shy = Label(root, text="shift y by:").grid(row=6,column=1)
        entry_shy = Entry(root, textvariable=shift_y).grid(row=7, column=1)

        shift_z = DoubleVar()
        label_shz = Label(root, text="shift z by:").grid(row=6,column=2)
        entry_shz = Entry(root, textvariable=shift_z).grid(row=7, column=2)

        button_shx = Button(root, text="shift", command=lambda:self.shift(shift_x.get(), shift_y.get(), shift_z.get())).grid(row=6, column=3)

        autofit = Button(root, text="autofit", command=lambda:self.autofit()).grid(row=8, column=1)
        autoscale = Button(root, text="autoscale", command=lambda:self.autofit(rotate=True, translate=False, scale=3)).grid(row=8, column=2)
        reset = Button(root, text="reset", command=lambda:self.reset()).grid(row=8, column=0)

        root.mainloop() # start the gui

    def rotate_x(self, alpha):
        Rx = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)] ])
        self.points = np.matmul(self.points, Rx)
        self.update_plot()

    def rotate_y(self, alpha):
        Ry = np.array([[np.cos(alpha), 0, np.sin(alpha)],[0, 1, 0],[-np.sin(alpha), 0, np.cos(alpha)]])
        self.points = np.matmul(self.points, Ry)
        self.update_plot()

    def rotate_z(self, alpha):
        Rz = np.array([[np.cos(alpha), np.sin(alpha), 0],[-np.sin(alpha), np.cos(alpha), 0],[0, 0, 1]])
        self.points = np.matmul(self.points, Rz)
        self.update_plot()

    def scale(self, sx, sy, sz):
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
        self.points = np.matmul(self.points, S)
        self.update_plot()

    def scale_x(self, fx):
        Sx = np.array([[fx, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.points = np.matmul(self.points, Sx)
        self.update_plot()

    def scale_y(self, fy):
        Sy = np.array([[1, 0, 0], [0, fy, 0], [0, 0, 1]])
        self.points = np.matmul(self.points, Sy)
        self.update_plot()

    def scale_z(self, fz):
        Sz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, fz]])
        self.points = np.matmul(self.points, Sz)
        self.update_plot()

    def shift(self, sx, sy, sz):
        self.points += np.array([sx/1000, sy/1000, sz/1000], dtype=float)
        self.update_plot()


    def autofit(self, rotate=True, translate=True, scale=0):
        trans = fit_point_cloud(self.points, self.surface["rr"], rotate=rotate,
            translate=translate, scale=scale, x0=None, leastsq_args={}, out="trans")
        ones =np.expand_dims(np.ones(len(self.points)),axis=1)
        self.points = np.matmul(np.hstack((self.points,ones)),trans)[:,0:3]
        self.update_plot()

    def reset(self):
        self.load_electrodes()
        self.points = self.electrodes
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
        surface = mlab.pipeline.surface(mesh, color=(1,1,1),figure=self.fig, opacity=0.5) # Plot bem surface --> see mne.viz._3d._create_mesh_surf
        self.electrode_points = mlab.points3d(self.electrodes[:, 0], self.electrodes[:, 1], self.electrodes[:, 2],scale_factor=0.01, figure=self.fig) # plot points
        self.nasion_point = mlab.points3d(self.nasion[0], self.nasion[1], self.nasion[2],scale_factor=0.01, figure=self.fig, color=(0,0,1)) # plot points

    def update_plot(self):

        if self.pts_type == "electrodes":
            self.electrodes = self.points
        if self.pts_type =="nasion":
            self.nasion = self.points
        # # use distance for color and size coding
        self.electrode_points.remove()
        self.nasion_point.remove()
        self.electrode_points = mlab.points3d(self.electrodes[:, 0], self.electrodes[:, 1], self.electrodes[:, 2],scale_factor=0.01, figure=self.fig) # plot points
        self.nasion_point = mlab.points3d(self.nasion[0], self.nasion[1], self.nasion[2],scale_factor=0.01, figure=self.fig, color=(0,0,1)) # plot points
        distance = np.linalg.norm(np.abs(self.electrodes[-1]-self.nasion))*1000
        print("distance from ground to nasion: %s mm" % (distance))

    def drop_electrodes(self):

        dist, ind = self.tgt_tree.query(self.electrodes)
        self.electrodes = self.surface["rr"][ind]
        self.electrodes = np.concatenate([x for x in self.electrodes])

    def make_montage(self, plot=True, save=False):

        dig_ch_pos = dict(zip(self.ch_names, self.electrodes[:64]))
        montage = DigMontage(nasion=self.nasion, point_names=self.ch_names, dig_ch_pos=dig_ch_pos)
        # saving montage changes names, just save positions as .npy
        if plot:
            montage.plot(show_names=True, kind="topomap")
        if save:
            np.save(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_montage.npy"),self.electrodes[:64])
            np.save(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_nasion.npy"),self.nasion)
            np.save(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_ground.npy"),self.electrodes[-1])
            np.save(os.path.join(os.environ["EXPDIR"],os.environ["SUBJECT"],os.environ["SUBJECT"]+"_reference.npy"),self.electrodes[-2])


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.environ["PYDIR"])
    from scale_electrodes import *
    a = align_electrodes(subject="eegl01", capsize=56)
