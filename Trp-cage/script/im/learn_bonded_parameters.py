#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=learn_bonded
#SBATCH --time=10:00:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-11
#SBATCH --mem=100G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/learn_bonded_%a.txt

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
import pandas as pd

protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 1
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)
    
data = torch.load(f"./output/{name}/md/ic.pt")
ic = data['ic']
ic.double()

ic.angle[ic.angle <= 0] = 0.0
ic.angle[ic.angle >= math.pi] = math.pi
ic.dihedral[ic.dihedral <= -math.pi] = math.pi
ic.dihedral[ic.dihedral >= math.pi] = math.pi

# T = 300 * unit.kelvin
# kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA
# kbT_kJ_per_mole = kbT.value_in_unit(unit.kilojoule_per_mole)
# beta = 1/kbT_kJ_per_mole

bonded_parameters = {}
bonded_parameters['reference_particle_2_bond'] = {
    'b0': torch.mean(ic.reference_particle_2_bond).item(),
    'kb': 1/torch.var(ic.reference_particle_2_bond).item()
}

bonded_parameters['reference_particle_3_bond'] = {
    'b0': torch.mean(ic.reference_particle_3_bond).item(),
    'kb': 1/torch.var(ic.reference_particle_3_bond).item()
}

bonded_parameters['bond'] = {
    'b0': torch.mean(ic.bond, 0).numpy(),
    'kb': 1/torch.var(ic.bond, 0).numpy()
}

angle_md = torch.cat([ic.reference_particle_3_angle[:, None], ic.angle], dim = -1).numpy()

angle_parameters = []

os.makedirs(f"./output/{name}/im/FF", exist_ok = True)
pdf = PdfPages(f"./output/{name}/im/FF/angle_hist_from_md_and_cl.pdf")

with open(f"./output/{name}/md/basis_knots_angle_and_dihedrals.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
angle_knots = data['angle_knots']
angle_boundary_knots = data['angle_boundary_knots']

dihedral_knots = data['dihedral_knots']
dihedral_boundary_knots = data['dihedral_boundary_knots']

for j in range(angle_md.shape[-1]):
    basis_md = bs(angle_md[:,j], angle_knots, angle_boundary_knots)
    basis_noise = bs(np.random.uniform(0, math.pi, angle_md.shape[0]),
                      angle_knots, angle_boundary_knots)
    basis_md = torch.from_numpy(basis_md)
    basis_noise = torch.from_numpy(basis_noise)
    
    log_q_md = basis_md.new_ones(basis_md.shape[0])*np.log(1./np.pi) 
    log_q_noise = basis_noise.new_ones(basis_noise.shape[0])*np.log(1./np.pi)
    
    alphas, F = contrastive_learning(log_q_noise, log_q_md,
                                     basis_noise, basis_md)
    
    a_over_the_range = np.linspace(0.0, math.pi, 1000)    
    basis_over_the_range = bs(a_over_the_range, angle_knots, angle_boundary_knots)
    U = np.matmul(basis_over_the_range, alphas)
    
    fig = plt.figure(0)
    fig.clf()
    axes = plt.hist(angle_md[:,j], bins = 50, range = (0.0, np.pi),
                    density = True,
                    log = False, label = 'md')
    p = np.exp(-(U - F))
    p = p*axes[0].max()/p.max()

    plt.plot(a_over_the_range, p, '-', label = 'cl', linewidth = 4)
    plt.legend()
    pdf.savefig()

    angle_parameters.append({'alphas': alphas,
                             'F': F,
                             'U': U,
                             'a_over_the_range': a_over_the_range})

    
pdf.close()
bonded_parameters['reference_particle_3_angle'] = angle_parameters[0]
bonded_parameters['angle'] = angle_parameters[1:]

dihedral_md = ic.dihedral.numpy()
dihedral_parameters = []

os.makedirs(f"./output/{name}/im/FF", exist_ok = True)
pdf = PdfPages(f"./output/{name}/im/FF/dihedral_hist_from_md_and_cl.pdf")

for j in range(dihedral_md.shape[-1]):
    basis_md = pbs(dihedral_md[:,j], dihedral_knots, dihedral_boundary_knots)
    basis_noise = pbs(np.random.uniform(-math.pi, math.pi, dihedral_md.shape[0]),
                      dihedral_knots, dihedral_boundary_knots)
    basis_md = torch.from_numpy(basis_md)
    basis_noise = torch.from_numpy(basis_noise)
    
    log_q_md = basis_md.new_ones(basis_md.shape[0])*np.log(1./(2*np.pi)) 
    log_q_noise = basis_noise.new_ones(basis_noise.shape[0])*np.log(1./(2*np.pi))
    
    alphas, F = contrastive_learning(log_q_noise, log_q_md,
                                     basis_noise, basis_md)

    d_over_the_range = np.linspace(-math.pi, math.pi, 1000)    
    basis_over_the_range = pbs(d_over_the_range, dihedral_knots, dihedral_boundary_knots)
    U = np.matmul(basis_over_the_range, alphas)
    
    fig = plt.figure(0)
    fig.clf()
    axes = plt.hist(dihedral_md[:,j], bins = 50, range = (-np.pi, np.pi),
                    density = True,
                    log = False, label = 'md')
    p = np.exp(-(U - F))
    p = p*axes[0].max()/p.max()

    plt.plot(d_over_the_range, p, '-', label = 'cl', linewidth = 4)
    plt.legend()
    pdf.savefig()

    dihedral_parameters.append({'alphas': alphas,
                                'F': F,
                                'U': U,
                                'd_over_the_range': d_over_the_range})
    
pdf.close()
bonded_parameters['dihedral'] = dihedral_parameters

os.makedirs(f"./output/{name}/im/FF", exist_ok = True)
with open(f"./output/{name}/im/FF/bonded_parameters.pkl", 'wb') as file_handle:
    pickle.dump(bonded_parameters,
                file_handle)
exit()
