import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
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

exit()

fig = plt.figure(np.random.randint(0, 1000), figsize = (6.4*4, 4.8))
fig.clf()
plt.subplot(1,4,1)
n_md, bins, _ = plt.hist(rmsd_md, bins = 26, density = True, alpha = 0.6, label = 'All atom', log = False, color = 'C1')
n_cg_lj, _, _ = plt.hist(rmsd_cg_lj, bins = bins, density = True, alpha = 0.6, label = 'CG (pairwise)', log = False, color = 'C2')
# plt.xlabel("RMSD (nm)")
# plt.ylabel("Probability density")
plt.ylim(0, 4.0)
plt.legend()

plt.subplot(1,4,2)
n_md, bins, _ = plt.hist(rmsd_md, bins = bins, density = True, alpha = 0.6, label = 'All atom', log = False, color = 'C1')
n_cg_rmsd, _, _ = plt.hist(rmsd_cg_rmsd, bins = bins, density = True, alpha = 0.6, label = 'CG (pairwise + mb-rmsd)', log = False, color = 'C0')
# plt.xlabel("RMSD (nm)")
# plt.ylabel("Probability density")
plt.ylim(0, 4.0)
plt.legend()

plt.subplot(1,4,3)
n_md, bins, _ = plt.hist(rmsd_md, bins = bins, density = True, alpha = 0.6, label = 'All atom', log = False, color = 'C1')
n_cg_nn, _, _ = plt.hist(rmsd_cg_nn, bins = bins, density = True, alpha = 0.6, label = 'CG (pairwise + mb-network)', log = False, color = 'k')
# plt.xlabel("RMSD (nm)")
# plt.ylabel("Probability density")
plt.ylim(0, 4.0)
plt.legend()

plt.subplot(1,4,4)
F_md = -np.log(n_md)
F_cg_lj = -np.log(n_cg_lj)
F_cg_rmsd = -np.log(n_cg_rmsd)
F_cg_nn = -np.log(n_cg_nn)

plt.plot(0.5*(bins[1:] + bins[0:-1]), F_md, linewidth = 2, label = "All atom", color = 'C1')
plt.plot(0.5*(bins[1:] + bins[0:-1]), F_cg_lj, linewidth = 2, label = "CG (pairwise)", color = 'C2')
plt.plot(0.5*(bins[1:] + bins[0:-1]), F_cg_rmsd, linewidth = 2, label = "CG (pairwise + mb-rmsd)", color = 'C0')
plt.plot(0.5*(bins[1:] + bins[0:-1]), F_cg_nn, linewidth = 2, label = "CG (pairwise + mb-network])", color = 'k')
# plt.xlabel("RMSD (nm)")
# plt.ylabel("Free energy (kT)")
#plt.xlim(0, 1.8)
plt.ylim(None, 11.1)
plt.legend()
#plt.tight_layout()
#n, bins = np.histogram(rmsd_md, bins = 30, density = True)

os.makedirs("./output/plots", exist_ok = True)
fig.savefig("./output/plots/2JOF_rmsd_hist.eps")
fig.savefig("./output/plots/2JOF_rmsd_hist.pdf")

exit()
