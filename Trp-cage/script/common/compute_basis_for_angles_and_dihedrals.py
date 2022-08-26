#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=basis_angle_dihedral
#SBATCH --time=00:30:00
#SBATCH --partition=xeon-p8
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

protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 1
name = protein_names.iloc[job_idx, 0]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str)
args = parser.parse_args()
model = args.model

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)
data = torch.load(f"./output/{name}/{model}/ic.pt")
ic = data['ic']
    
ic.double()
ic.reference_particle_3_angle[ic.reference_particle_3_angle <= 0] = 0.0
ic.reference_particle_3_angle[ic.reference_particle_3_angle >= math.pi] = math.pi

ic.angle[ic.angle <= 0] = 0.0
ic.angle[ic.angle >= math.pi] = math.pi

angle_knots = np.linspace(0, math.pi, 20, endpoint = False)[1:]
angle_boundary_knots = np.array([0.0, math.pi])

basis = {}
basis['reference_particle_3_angle'] = bs(ic.reference_particle_3_angle.numpy(), angle_knots, angle_boundary_knots)
basis['angle'] = np.stack([bs(ic.angle[:,j].numpy(), angle_knots, angle_boundary_knots) for j in range(ic.angle.shape[1])], axis = 1)

angle_over_range = np.linspace(0, math.pi, 10000)
basis_over_range = {}
basis_over_range['angle'] = bs(angle_over_range, angle_knots, angle_boundary_knots)

ic.dihedral[ic.dihedral <= -math.pi] = math.pi
ic.dihedral[ic.dihedral >= math.pi] = math.pi

dihedral_knots = np.linspace(-math.pi, math.pi, 20, endpoint = False)[1:]
dihedral_boundary_knots = np.array([-math.pi, math.pi])
basis['dihedral'] = np.stack([pbs(ic.dihedral[:,j].numpy(), dihedral_knots, dihedral_boundary_knots) for j in range(ic.dihedral.shape[1])], axis = 1)

dihedral_over_range = np.linspace(-math.pi, math.pi, 10000)
basis_over_range['dihedral'] = pbs(dihedral_over_range, dihedral_knots, dihedral_boundary_knots)

with open(f"./output/{name}/{model}/basis_knots_angle_and_dihedrals.pkl", 'wb') as file_handle:
    pickle.dump({'angle_knots': angle_knots,
                 'angle_boundary_knots': angle_boundary_knots,
                 'dihedral_knots': dihedral_knots,
                 'dihedral_boundary_knots': dihedral_boundary_knots},
                file_handle)

with open(f"./output/{name}/{model}/basis_angle_and_dihedrals.pkl", 'wb') as file_handle:
    pickle.dump({'basis': basis,
                 'basis_over_range': basis_over_range,
                 'angle_over_range': angle_over_range,
                 'dihedral_over_range': dihedral_over_range,
                 'angle_knots': angle_knots,
                 'angle_boundary_knots': angle_boundary_knots,
                 'dihedral_knots': dihedral_knots,
                 'dihedral_boundary_knots': dihedral_boundary_knots},
                file_handle)

with open(f"./output/{name}/{model}/basis_over_range_angle_and_dihedrals.pkl", 'wb') as file_handle:
    pickle.dump({'basis_over_range': basis_over_range,
                 'angle_over_range': angle_over_range,
                 'dihedral_over_range': dihedral_over_range,
                 'angle_knots': angle_knots,
                 'angle_boundary_knots': angle_boundary_knots,
                 'dihedral_knots': dihedral_knots,
                 'dihedral_boundary_knots': dihedral_boundary_knots},
                file_handle)
    
    
    
    

