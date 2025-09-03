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
import pickle

name = '2JOF'

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)

traj_cg_lj = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_False_weight_decay_{4e-7:.3E}.dcd", psf)
traj_cg_rmsd = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_True_weight_decay_{2e-7:.3E}.dcd", psf)
traj_cg_nn = mdtraj.load_dcd(f"./output/{name}/nnforce_NVT/full_rmsd_False_weight_decay_4e-07_weight_decay_{1e-3:.3E}.dcd", psf)

traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
num_residues = psf.n_residues
indices = []
for i in range(num_residues):
    for j in range(i+6, num_residues):
        indices.append([i,j])

distances_md = mdtraj.compute_distances(traj_md, np.array(indices))
distances_cg_lj = mdtraj.compute_distances(traj_cg_lj, np.array(indices))
distances_cg_rmsd = mdtraj.compute_distances(traj_cg_rmsd, np.array(indices))
distances_cg_nn = mdtraj.compute_distances(traj_cg_nn, np.array(indices))

nrow = 15
ncol = 7

fig = plt.figure(np.random.randint(0, 1000), figsize = (6.4*ncol, 4.8*nrow))
fig.clf()

for j in range(len(indices)):
    print(j)
    ax = plt.subplot(nrow, ncol, j+1)
    idx1, idx2 = indices[j]
    plt.hist(distances_md[:, j], bins = 100, density = True, alpha = 0.5, label = 'All atom', range = [0.0, 2.5], log = False, color = 'C1')
    plt.hist(distances_cg_nn[:, j], bins = 100, density = True, alpha = 0.5, label = r'CG (pairwise + mb-nn)', range = [0.0, 2.5], log = False, color = 'k')
    plt.title(f"{idx1}-{idx2}")
#    plt.yscale('log')
    plt.legend()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc = (0.29, 0.13))

plt.tight_layout()
fig.savefig(f"./output/plots/{name}_distances_nn.pdf")
plt.close()

exit()

## load spline coefficients
with open(f"./output/LJ/{name}/FF_learned_with_{noise_model}_as_noise/weight_decay_{weight_decay:.3E}.pkl", 'rb') as file_handle:
    FF = pickle.load(file_handle)
    LJ_parameters = FF['LJ_parameters']

pair_indices = list(LJ_parameters.keys())
pair_indices.sort()

nrow = 6
ncol = 4

fig = plt.figure(np.random.randint(0, 1000), figsize = (6.4*ncol, 4.8*nrow))
fig.clf()
for k in range(len(pair_indices)):
    plt.subplot(nrow, ncol, k+1)
    pair = pair_indices[k]
    i,j = pair_indices[k]

    aa1, aa2 = resnames[i], resnames[j]
    if resnames[j] < resnames[i]:
        aa1, aa2 = resnames[j], resnames[i]
    
    r_min, r_max = LJ_parameters[pair]['r_min'], LJ_parameters[pair]['r_max']
    ulj = LJ_parameters[pair]['ulj']
    r = np.linspace(r_min , r_max, len(ulj))
    plt.plot(r, ulj, linewidth = 6.0)
    plt.xlim(0, r_max)
    plt.ylim(ulj.min()-1, 7)

    plt.title(f"{i}-{j}:{aa1}-{aa2}")
    print(k)
    
fig.savefig(f"./output/plots/{name}_lj.eps")

exit()
