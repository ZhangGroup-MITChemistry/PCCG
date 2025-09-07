import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
import numpy as np
import torch
import pyemma
import mdtraj
import os
from scipy.stats import gaussian_kde
from sys import exit
from matplotlib import cm

name = "2JOF"

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)

traj_cg_lj = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_False_weight_decay_{4e-7:.3E}.dcd", psf)
traj_cg_rmsd = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_True_weight_decay_{2e-7:.3E}.dcd", psf)
traj_cg_nn = mdtraj.load_dcd(f"./output/{name}/nnforce_NVT/full_rmsd_False_weight_decay_4e-07_weight_decay_{1e-3:.3E}.dcd", psf)

traj_reference = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)

rmsd_md = mdtraj.rmsd(traj_md, traj_reference, 0)
rmsd_cg_lj = mdtraj.rmsd(traj_cg_lj, traj_reference, 0)
rmsd_cg_rmsd = mdtraj.rmsd(traj_cg_rmsd, traj_reference, 0)
rmsd_cg_nn = mdtraj.rmsd(traj_cg_nn, traj_reference, 0)

rmsd_max = max(rmsd_md.max(), rmsd_cg_lj.max(), rmsd_cg_rmsd.max(), rmsd_cg_nn.max())

N = 3000
rmsd_md = rmsd_md[::len(rmsd_md)//N]
rmsd_cg_lj = rmsd_cg_lj[::len(rmsd_cg_lj)//N]
rmsd_cg_rmsd = rmsd_cg_rmsd[::len(rmsd_cg_rmsd)//N]
rmsd_cg_nn = rmsd_cg_nn[::len(rmsd_cg_nn)//N]

fig = plt.figure(figsize = (6.4*4, 4.8*2))
fig.clf()
plt.subplot(4,1,1)
plt.plot(np.linspace(0, 208_000, len(rmsd_md)), rmsd_md, '.', label = 'All atom', ms = 6)
plt.ylim(0, rmsd_max)
plt.xlabel('Simulation time (ns)')
plt.ylabel('RMSD (nm)')

plt.subplot(4,1,2)
plt.plot(np.linspace(0, 500, len(rmsd_cg_lj)), rmsd_cg_lj, '.', label = 'CG (pairwise)', ms = 5)
plt.ylim(0, rmsd_max)
plt.xlabel('Simulation time (ns)')
plt.ylabel('RMSD (nm)')

plt.subplot(4,1,3)
plt.plot(np.linspace(0, 500, len(rmsd_cg_rmsd)), rmsd_cg_rmsd, '.', label = 'CG (pairse + mb-rmsd)', ms = 5)
plt.ylim(0, rmsd_max)
plt.xlabel('Simulation time (ns)')
plt.ylabel('RMSD (nm)')

plt.subplot(4,1,4)
plt.plot(np.linspace(0, 500, len(rmsd_cg_nn)), rmsd_cg_nn, '.', label = 'CG (pairse + mb-network)', ms = 5)
plt.ylim(0, rmsd_max)
plt.xlabel('Simulation time (ns)')
plt.ylabel('RMSD (nm)')

#plt.legend()
plt.tight_layout()
fig.savefig(f"./output/plots/{name}_rmsd_traj.eps")

exit()
