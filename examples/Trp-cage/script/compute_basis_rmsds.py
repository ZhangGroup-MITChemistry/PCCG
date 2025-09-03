#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=basis_rmsd
#SBATCH --time=00:30:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --array=0-1
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/basis_rmsd_%a.txt

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import pandas as pd
from sys import exit
import simtk.unit as unit
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import os
import argparse
import mdtraj
import simtk.unit as unit
import math
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
from MMFlow import utils
from CLCG.utils.splines import *
from CLCG.utils.CL import *
from scipy.sparse import csr_matrix
import pandas as pd
import os
import itertools

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

traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)

ss = pd.read_table("./info/secondary_structure.csv",
                   index_col = 'name', header = 0)
ss = eval(ss.loc[name, 'ss'])

if len(ss) == 0:
    exit()

basis = []
    
for idx_ss in range(len(ss)):
    particle_index = [list(range(start-1, end)) for start, end in ss[idx_ss]]
    particle_index = list(itertools.chain(*particle_index))

    rmsd_md = mdtraj.rmsd(traj_md, traj_ref, atom_indices = particle_index)
    rmsd_imus = mdtraj.rmsd(traj_imus, traj_ref, atom_indices = particle_index)

    rmsd_md = torch.from_numpy(rmsd_md).double()
    rmsd_imus = torch.from_numpy(rmsd_imus).double()

    rmsd_max = rmsd_md.max().item()
    num_of_basis = 15

    exit()
    
    basis_md = bs_rmsd(rmsd_md, rmsd_max, num_of_basis)
    basis_imus = bs_rmsd(rmsd_imus, rmsd_max, num_of_basis)

    rmsd_grid = torch.linspace(0, rmsd_max, steps = 1000)
    basis_grid = {}
    basis_grid['basis'] = bs_rmsd(rmsd_grid, rmsd_max, num_of_basis)
    basis_grid['min'] = 0.0
    basis_grid['max'] = rmsd_max

    basis.append({'md': basis_md,
                  'imus': basis_imus,
                  'grid': basis_grid})

torch.save(basis, f"./output/{name}/basis_rmsd.pt")
    
exit()    
    
    

