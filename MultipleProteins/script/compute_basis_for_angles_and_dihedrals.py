#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=basis_angle_dihedral
#SBATCH --time=01:00:00
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --array=0-11
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/basis_angle_dihedral_%a.txt

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

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
with open(f"./output/{name}/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)
    
data = torch.load(f"./output/{name}/ic_md.pt")
ic_md = data['ic']

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

xyz_imus = traj_imus.xyz.astype(np.float64)
ic_imus, ic_logabsdet = coor_transformer.compute_internal_coordinate_from_xyz(torch.from_numpy(xyz_imus))
torch.save({'ic': ic_imus, 'ic_logabsdet': ic_logabsdet},
           f"./output/{name}/ic_imus.pt")

ic_md.double()
ic_imus.double()

ic_md.reference_particle_3_angle.clamp_(0.0, math.pi)
ic_md.angle.clamp_(0.0, math.pi)
ic_md.dihedral.clamp_(-math.pi, math.pi)

ic_imus.reference_particle_3_angle.clamp_(0.0, math.pi)
ic_imus.angle.clamp_(0.0, math.pi)
ic_imus.dihedral.clamp_(-math.pi, math.pi)

data = torch.load(f"./output/{name}/knots_for_angle_and_dihedral.pt")
angle_knots = data['angle_knots']
angle_boundary_knots = data['angle_boundary_knots']

dihedral_knots = data['dihedral_knots']
dihedral_boundary_knots = data['dihedral_boundary_knots']

basis_md = {}
basis_md['reference_particle_3_angle'] = bs(ic_md.reference_particle_3_angle, angle_knots, angle_boundary_knots)
basis_md['angle'] = torch.cat(
    [bs(ic_md.angle[:,j], angle_knots, angle_boundary_knots) for j in range(ic_md.angle.shape[1])],
    dim = -1)

basis_md['dihedral'] = torch.cat(
    [pbs(ic_md.dihedral[:,j],
         dihedral_knots,
         dihedral_boundary_knots)
     for j in range(ic_md.dihedral.shape[1])],
    dim = -1)

basis_imus = {}
basis_imus['reference_particle_3_angle'] = bs(ic_imus.reference_particle_3_angle,
                                              angle_knots,
                                              angle_boundary_knots)
basis_imus['angle'] = torch.cat(
    [bs(ic_imus.angle[:,j],
        angle_knots,
        angle_boundary_knots)
     for j in range(ic_imus.angle.shape[1])],
    dim = -1)
basis_imus['dihedral'] = torch.cat(
    [pbs(ic_imus.dihedral[:,j],
         dihedral_knots,
         dihedral_boundary_knots)
     for j in range(ic_imus.dihedral.shape[1])],
    dim = -1)

# for k in basis_md.keys():
#     basis_md[k] = basis_md[k].to_sparse()
#     basis_imus[k] = basis_imus[k].to_sparse()        

basis_grid = {}
basis_grid['angle_min'] = 0.0
basis_grid['angle_max'] = math.pi
angle_grid = torch.linspace(basis_grid['angle_min'], basis_grid['angle_max'], 1000)
basis_grid['angle_basis'] = bs(angle_grid,
                               angle_knots,
                               angle_boundary_knots)

basis_grid['dihedral_min'] = -math.pi
basis_grid['dihedral_max'] = math.pi
dihedral_grid = torch.linspace(basis_grid['dihedral_min'], basis_grid['dihedral_max'], 1000)
basis_grid['dihedral_basis'] = pbs(dihedral_grid,
                                   dihedral_knots,
                                   dihedral_boundary_knots)

torch.save({'md': basis_md, 'imus': basis_imus, 'grid': basis_grid},
           f'./output/{name}/basis_angle_and_dihedrals.pt')

exit()    
    
    

