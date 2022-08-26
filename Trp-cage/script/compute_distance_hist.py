#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=basis_lj
#SBATCH --time=00:30:00
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --array=0-1
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/basis_lj_%a.txt

import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad
import pickle
import mdtraj
from sys import exit
import os
import argparse
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
from CLCG.utils.splines import *
from CLCG.utils.CL import *
import torch
torch.set_default_dtype(torch.double)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import pandas as pd
import ray

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
job_idx = 1
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf, stride = 1)

with open(f"./output/{name}/rmsd_centers_and_k_for_imus.pkl", 'rb') as file_handle:
    rmsd_centers_and_k = pickle.load(file_handle)
size = rmsd_centers_and_k['size']
traj_imus = []
stride = 10
for rank in range(size):
    print(f"rank: {rank}")
    traj = mdtraj.load_dcd(f"./output/{name}/traj_imus/traj_{rank}.dcd", psf)
    traj = traj[::stride]
    traj_imus.append(traj)
traj_imus = mdtraj.join(traj_imus)

indices = [(i,j) for i in range(psf.n_residues) for j in range(i+4, psf.n_residues)]
r_md = mdtraj.compute_distances(traj_md, indices)
r_imus = mdtraj.compute_distances(traj_imus, indices)

ncol = 6
nrow = r_md.shape[-1]//ncol + 1

fig = plt.figure(figsize = (6.4*ncol, 4.8*nrow))
fig.clf()
for i in range(r_md.shape[1]):
    plt.subplot(nrow, ncol, i+1)
    plt.hist(r_md[:, j], bins = 100, density = True, alpha = 0.5, label = 'All atom', log = True)
    plt.hist(r_imus[:, j], bins = 100, density = True, alpha = 0.5, label = 'IMUS', log = True)
    plt.tight_layout()

os.makedirs(f"./output/LJ/{name}/CG_simulations/", exist_ok = True)
fig.savefig(f"./output/LJ/{name}/CG_simulations/pairwise_distances_hist_{noise_model}_weight_decay_{weight_decay:.3E}.pdf")
