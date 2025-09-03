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

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
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
num_of_basis = 12

r_md = mdtraj.compute_distances(traj_md, indices)
r_md = torch.from_numpy(r_md).double()

r_min, _ = torch.min(r_md, 0)
r_max = 1.5

r_imus = mdtraj.compute_distances(traj_imus, indices)
r_imus = torch.from_numpy(r_imus).double()

ray.init(ignore_reinit_error = True, _temp_dir = "/home/gridsan/dingxq/tmp/ray")

r_md = ray.put(r_md)
r_imus = ray.put(r_imus)

r_grid_min = 0.2
r_grid_max = r_max
r_grid = ray.put(torch.linspace(r_grid_min, r_grid_max, 1000))

@ray.remote
def compute_lj_basis(j):
    import sys
    sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
    from CLCG.utils.splines import bs_lj
    print(j)
    basis_md = bs_lj(ray.get(r_md)[:,j], r_min[j].item(), r_max, num_of_basis, omega = False)
    basis_imus = bs_lj(ray.get(r_imus)[:,j], r_min[j].item(), r_max, num_of_basis, omega = False)
    basis_grid, omega = bs_lj(ray.get(r_grid), r_min[j].item(), r_max, num_of_basis, omega = True)
    
    return basis_md, basis_imus, basis_grid, omega

res = ray.get([compute_lj_basis.remote(j) for j in range(len(indices))])

basis_md = torch.cat([ res[i][0] for i in range(len(res))], dim = 1)
basis_imus = torch.cat([ res[i][1] for i in range(len(res))], dim = 1)
basis_grid = {}
basis_grid['basis'] = torch.stack([ res[i][2] for i in range(len(res))])
basis_grid['min'] = r_grid_min
basis_grid['max'] = r_grid_max
omega = torch.stack([ res[i][3] for i in range(len(res))])
basis_grid['omega'] = omega

# basis_md = basis_md.to_sparse()
# basis_imus = basis_imus.to_sparse()

torch.save({'md': basis_md,
            'imus': basis_imus,
            'grid': basis_grid},
           f"./output/{name}/basis_lj.pt")
exit()
