#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=basis_lj
#SBATCH --time=1-00:00:00
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --array=0-11
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
from scipy.sparse import csr_matrix

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = -1
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

traj_md = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", psf, stride = 1)

with open(f"./output/{name}/rmsd_centers_and_k_for_imus.pkl", 'rb') as file_handle:
    rmsd_centers_and_k = pickle.load(file_handle)
size = rmsd_centers_and_k['size']
traj_imus = []
for rank in range(size):
    print(f"rank: {rank}")
    traj = mdtraj.load_dcd(f"./output/{name}/traj_imus/traj_{rank}.dcd", psf)
    traj_imus.append(traj)
traj_imus = mdtraj.join(traj_imus)
stride = size // 10
traj_imus = traj_imus[::stride]

with open(f"./output/common/LJ_rmin.pkl", 'rb') as file_handle:
    r_min_dict = pickle.load(file_handle)
aa_pairs = list(r_min_dict.keys())
aa_pairs.sort()

num_aa_pairs = len(aa_pairs)
num_of_basis = 12

resnames = [res.name for res in psf.residues]

def sort_pair_name(pair):
    if pair[0] <= pair[1]:
        return pair
    else:
        tmp = pair[0]
        pair[0] = pair[1]
        pair[1] = tmp
        return pair

pair_indices = [(i,j) for i in range(psf.n_residues)
                for j in range(i+4, psf.n_residues)]

r_md = mdtraj.compute_distances(traj_md, pair_indices)
r_imus = mdtraj.compute_distances(traj_imus, pair_indices)

r_min = []
for i,j in pair_indices:
    pair = tuple(sort_pair_name([resnames[i], resnames[j]]))    
    r_min.append(r_min_dict[pair])
r_min= np.array(r_min)
r_max = 1.5

ray.init(ignore_reinit_error = True, _temp_dir = "/home/gridsan/dingxq/tmp/ray")

r_md = ray.put(r_md)
r_imus = ray.put(r_imus)
r_min = ray.put(r_min)

@ray.remote
def compute_lj_basis(j, r_md, r_imus, r_min):
    import sys
    sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
    from CLCG.utils.splines import bs_lj
    print(j)

    r = torch.from_numpy(np.copy(r_md[:,j]))
    basis_md = bs_lj(r, r_min[j], r_max, num_of_basis, omega = False)

    r = torch.from_numpy(np.copy(r_imus[:,j]))    
    basis_imus = bs_lj(r, r_min[j], r_max, num_of_basis, omega = False)

    return basis_md, basis_imus

res = ray.get([compute_lj_basis.remote(j, r_md, r_imus, r_min) for j in range(len(pair_indices))])

basis_md = torch.zeros((len(traj_md), num_of_basis*num_aa_pairs))
basis_imus = torch.zeros((len(traj_imus), num_of_basis*num_aa_pairs))

# basis_md = csr_matrix((len(traj_md), num_of_basis*num_aa_pairs))
# basis_imus = csr_matrix((len(traj_imus), num_of_basis*num_aa_pairs))

for k in range(len(pair_indices)):
    i,j = pair_indices[k]
    print(i,j, flush = True)
    pair = tuple(sort_pair_name([resnames[i], resnames[j]]))

    # for md 
    basis = res[k][0]
    idx = aa_pairs.index(pair)
    basis_md[:, idx*num_of_basis:(idx+1)*num_of_basis] += basis
    
    # data = basis.data
    # indices = basis.indices
    # indptr = basis.indptr
    # indices = indices + num_of_basis*aa_pairs.index(pair)
    # basis = csr_matrix((data, indices, indptr), shape = basis_md.shape)
    # basis_md = basis_md + basis

    ## for imus
    basis = res[k][1]
    basis_imus[:, idx*num_of_basis:(idx+1)*num_of_basis] += basis
    
    # data = basis.data
    # indices = basis.indices
    # indptr = basis.indptr
    # indices = indices + num_of_basis*aa_pairs.index(pair)
    # basis = csr_matrix((data, indices, indptr), shape = basis_imus.shape)
    # basis_imus = basis_imus + basis

# def scipy_csr_to_torch_coo(s):
#     s = s.tocoo()
#     s = torch.sparse_coo_tensor(np.array([s.row, s.col]),
#                                 s.data,
#                                 size = s.shape)
#     return s

# basis_md = scipy_csr_to_torch_coo(basis_md)
# basis_imus = scipy_csr_to_torch_coo(basis_imus)

torch.save({'md': basis_md,
            'imus': basis_imus},
           f"./output/{name}/basis_lj.pt")

exit()
if name != "CLN025":
    exit()
    
## for grid
basis_grid = {}
basis_grid['min'] = 0.1
basis_grid['max'] = 1.5
basis_grid['basis'] = []
basis_grid['omega'] = []

r_grid_min, r_grid_max = basis_grid['min'], basis_grid['max']
r_grid = torch.from_numpy(np.linspace(r_grid_min, r_grid_max, 1000))
r_grid = ray.put(r_grid)

@ray.remote
def compute_lj_basis_grid(r_grid, r_min):
    import sys
    sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
    from CLCG.utils.splines import bs_lj
    
    basis, omega = bs_lj(r_grid, r_min, r_max, num_of_basis, omega = True)
    return basis, omega

res = ray.get([compute_lj_basis_grid.remote(r_grid, r_min_dict[(aa1, aa2)]) for aa1, aa2 in aa_pairs])

basis_grid['basis'] = torch.stack([res[i][0] for i in range(len(res))])
basis_grid['omega'] = torch.stack([res[i][1] for i in range(len(res))])

torch.save(basis_grid, f"./output/common/basis_lj_grid.pt")
exit()
