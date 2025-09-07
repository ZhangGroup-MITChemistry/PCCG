#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=NVT_LJ
#SBATCH --time=10:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-8
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/NVT_LJ_%a.txt

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
import itertools
import torch
import pickle

name = "2JOF"
# include_rmsd = True
# weight_decay = 2e-7

include_rmsd = False
weight_decay = 4e-7

print(f"name: {name}, rmsd: {include_rmsd}, weight_decay: {weight_decay}", flush = True)

## read system
with open(f"./output/{name}/full_system/rmsd_{include_rmsd}_weight_decay_{weight_decay:.3E}.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

protein_info = pd.read_csv(
    "./info/protein_temperature_and_ionic_strength.txt",
    index_col = 'name',
    comment = '#')

## make an integrator
T = protein_info.loc[name, 'temperature']
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.kelvin * unit.AVOGADRO_CONSTANT_NA

fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

## read topology and pdb file
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
#traj_full = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_{include_rmsd}_weight_decay_{weight_decay:.3E}.dcd", psf)

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
xyz_init = traj_ref.xyz[0]

## minimize initial positions
platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)

energy_md = []
for i in range(traj_md.n_frames):
    context.setPositions(traj_md.xyz[i])
    state = context.getState(getEnergy = True)
    potential_energy = state.getPotentialEnergy()
    energy_md.append(potential_energy / kbT)

    if (i + 1) % 10000 == 0:
        print(i)
        
energy_imus = []
for i in range(traj_imus.n_frames):
    context.setPositions(traj_imus.xyz[i])
    state = context.getState(getEnergy = True)
    potential_energy = state.getPotentialEnergy()
    energy_imus.append(potential_energy / kbT)

    if (i + 1) % 10000 == 0:
        print(i)
    
energy_md = torch.tensor(energy_md)
energy_imus = torch.tensor(energy_imus)

os.makedirs(f"./output/{name}", exist_ok = True)
torch.save({'md': energy_md,
            'imus': energy_imus},
           f"./output/{name}/full_energy_rmsd_{include_rmsd}_weight_decay_{weight_decay:.3E}.pt")
