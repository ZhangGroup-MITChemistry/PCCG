#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=basis_LJ
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-30
#SBATCH --output=./slurm_output/basis_LJ_%a.txt
#SBATCH --open-mode=truncate

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

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
args = parser.parse_args()

name = args.name

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf, stride = 1)

size = 48
traj_im_us = []
for rank in range(size):
    traj = mdtraj.load_dcd(f"./output/{name}/im_us/trajs/traj_{rank}.dcd", psf, stride = 1)
    traj_im_us.append(traj)
    print(rank)
traj_im_us = mdtraj.join(traj_im_us)    

traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)

helix_particle_index = list(range(2,14))
rmsd_md = mdtraj.rmsd(traj_md, traj_ref, atom_indices = helix_particle_index)
rmsd_im_us = mdtraj.rmsd(traj_im_us, traj_ref, atom_indices = helix_particle_index)

rmsd_max = max(rmsd_md.max(), rmsd_im_us.max())
boundary_knots = [0, rmsd_max]
internal_knots = np.linspace(0, rmsd_max, num = 15, endpoint = False)[1:]

basis_md = bs(rmsd_md, internal_knots, boundary_knots)
basis_im_us = bs(rmsd_im_us, internal_knots, boundary_knots)

rmsd_over_range = np.linspace(0, rmsd_max, 1000, endpoint = True)
basis_over_range = bs(rmsd_over_range, internal_knots, boundary_knots)

with open(f"./output/{name}/md/basis_rmsd.pkl", 'wb') as file_handle:
    pickle.dump(basis_md, file_handle)

with open(f"./output/{name}/im_us/basis_rmsd.pkl", 'wb') as file_handle:
    pickle.dump(basis_im_us, file_handle)
    
with open(f"./output/{name}/md/basis_rmsd_over_range.pkl", 'wb') as file_handle:
    pickle.dump({'rmsd_over_range': rmsd_over_range,
                 'basis_over_range': basis_over_range},
                file_handle)
