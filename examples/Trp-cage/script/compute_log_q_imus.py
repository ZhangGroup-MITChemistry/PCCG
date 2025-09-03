#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=log_q_imus
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --array=0-1
#SBATCH --mem=60G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/log_q_imus_%a.txt

import numpy as np
import simtk.openmm.app  as app
import simtk.openmm as omm
import simtk.unit as unit
import argparse
import pandas as pd
import mdtraj
from sys import exit
import os
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from FastMBAR import *
import torch

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)

with open(f"./output/{name}/rmsd_centers_and_k_for_imus.pkl", 'rb') as file_handle:
    rmsd_centers_and_k = pickle.load(file_handle)
size = rmsd_centers_and_k['size']

traj_imus = []
num_frames = []
stride = 10
for rank in range(size):
    print(f"rank: {rank}")
    traj = mdtraj.load_dcd(f"./output/{name}/traj_imus/traj_{rank}.dcd", psf)
    traj = traj[::stride]
    traj_imus.append(traj)
    num_frames.append(len(traj))    
traj_imus = mdtraj.join(traj_imus)
num_frames = np.array(num_frames)

traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)

rmsd_md = mdtraj.rmsd(traj_md, traj_ref)
rmsd_imus = mdtraj.rmsd(traj_imus, traj_ref)

## make an integrator
protein_info = pd.read_csv("./info/protein_temperature_and_ionic_strength.txt",
                           index_col = 'name', comment = '#')
T = protein_info.loc[name, 'temperature']
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.kelvin * unit.AVOGADRO_CONSTANT_NA
kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)
platform = omm.Platform.getPlatformByName('Reference')
with open(f"./output/{name}/system_im.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)
context = omm.Context(system, integrator, platform)

energy_imus = []
for i in range(len(traj_imus)):
    context.setPositions(traj_imus.xyz[i])
    state = context.getState(getEnergy = True)
    U = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energy_imus.append(U)

    if (i + 1) % 10000 == 0:
        print(i, flush = True)
    
energy_imus = np.array(energy_imus)

energy_md = []
for i in range(len(traj_md)):
    context.setPositions(traj_md.xyz[i])
    state = context.getState(getEnergy = True)
    U = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energy_md.append(U)
energy_md = np.array(energy_md)

rmsd_centers = rmsd_centers_and_k['centers'][:, np.newaxis]
rmsd_k = rmsd_centers_and_k['k']

energy_matrix_imus = energy_imus + 0.5*rmsd_k*(rmsd_imus - rmsd_centers)**2
energy_matrix_md = energy_md + 0.5*rmsd_k*(rmsd_md - rmsd_centers)**2

energy_matrix_imus = energy_matrix_imus/kbT
energy_matrix_md = energy_matrix_md/kbT

fastmbar = FastMBAR(energy_matrix_imus, num_frames, verbose = True, cuda = True)
log_q_imus = fastmbar.log_prob_mix

biased_energy = energy_matrix_md + fastmbar.bias_energy.reshape((-1,1))
biased_energy_min = np.min(biased_energy, 0, keepdims = True)        
log_q_md = np.log(np.sum(np.exp(-(biased_energy - biased_energy_min)), 0)) - biased_energy_min.reshape(-1)

log_q_imus = torch.from_numpy(log_q_imus)
log_q_md = torch.from_numpy(log_q_md)

torch.save({'md': log_q_md, 'imus': log_q_imus},
           f"./output/{name}/log_q_imus.pt")
