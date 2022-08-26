#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=tica
#SBATCH --time=3:00:00
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --array=0-19
#SBATCH --mem=100G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/tica_%a.txt

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
import math
from sklearn.mixture import GaussianMixture
import ray
import copy

name = "2JOF"

weight_decay_list = [0.0,
                     1e-10, 5e-10,  
                     1e-9 , 5e-9 ,
                     1e-8 , 5e-8 ,
                     1e-7 , 5e-7 ,
                     1e-6 , 5e-6 ,
                     1e-5 , 5e-5 ,
                     1e-4 , 5e-4 ,
                     1e-3 , 5e-3 ,
                     1e-2 , 5e-2 ,
                     1e-1 ]

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
#weight_decay = weight_decay_list[job_idx]
weight_decay = 1e-3

# full_include_rmsd = False
# full_weight_decay = 4e-7

full_include_rmsd = False
full_weight_decay = 4e-7

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
traj_reference = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)

index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_md = mdtraj.compute_distances(traj_md, index)

tica = pyemma.coordinates.tica(distances_md, lag = 1000, dim = 2, kinetic_map = False)
x_md = tica.get_output()[0]

x = np.copy(x_md[x_md[:,1] < 5])
gm = GaussianMixture(n_components = 2)
gm.fit(x)

x = np.copy(x_md[(x_md[:,1] > 6) & (x_md[:,0] > 0)])
centers = np.concatenate([gm.means_, x.mean(0, keepdims = True)])
print(centers)

traj_md = traj_md.superpose(traj_reference, 0)
for i in range(centers.shape[0]):
    d = np.sqrt(np.sum((x_md - centers[i])**2, -1))
    idx = d.argsort()
    idx = idx[0:1]
    traj_md[idx].save_dcd(f"./output/plots/{name}_center_{i}.dcd")

exit()

flag_0 = x_md[:,0] < -1
flag_1 = (x_md[:,0] >= -1) & (x_md[:,1] < 5)
flag_2 = (x_md[:,0] >= -1) & (x_md[:,1] >= 5)

traj_md_0 = traj_md[flag_0]
traj_md_1 = traj_md[flag_1]
traj_md_2 = traj_md[flag_2]

traj_md_0_ref = ray.put(traj_md_0)
traj_md_1_ref = ray.put(traj_md_1)
traj_md_2_ref = ray.put(traj_md_2)

@ray.remote
def compute_rmsd(traj, i, j):
    traj = copy.deepcopy(traj)
    return [np.mean(mdtraj.rmsd(traj, traj, k)) for k in range(i,j) if k < traj.n_frames]

stride = 1000
rmsd_0 = ray.get([compute_rmsd.remote(traj_md_0_ref, i*stride, (i+1)*stride) for i in range(traj_md_0.n_frames//stride + 1)])
rmsd_1 = ray.get([compute_rmsd.remote(traj_md_1_ref, i*stride, (i+1)*stride) for i in range(traj_md_1.n_frames//stride + 1)])
rmsd_2 = ray.get([compute_rmsd.remote(traj_md_2_ref, i*stride, (i+1)*stride) for i in range(traj_md_2.n_frames//stride + 1)])

rmsd_0 = list(itertools.chain(*rmsd_0))
rmsd_1 = list(itertools.chain(*rmsd_1))
rmsd_2 = list(itertools.chain(*rmsd_2))

traj_md_0[np.argmin(rmsd_0)].save_xyz(f"./output/plots/{name}_center_0.xyz")
traj_md_1[np.argmin(rmsd_1)].save_xyz(f"./output/plots/{name}_center_1.xyz")
traj_md_2[np.argmin(rmsd_2)].save_xyz(f"./output/plots/{name}_center_2.xyz")

