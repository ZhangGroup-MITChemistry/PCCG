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

parser = argparse.ArgumentParser()
parser.add_argument('--rank', type = int)
parser.add_argument('--size', type = int)
args = parser.parse_args()

rank = args.rank
size = args.size
print(f"rank: {rank}, size: {size}", flush = True)

protein_names = pd.read_csv("./info/protein_names.txt",
                            comment = "#",
                            header = None)
protein_names = protein_names.iloc[0:2, 0].tolist()
flag_rmsd = [False, True]
# weight_decay_list = [1e-7, 5e-7,
#                      1e-6, 5e-6,
#                      1e-5, 5e-5,
#                      1e-4, 5e-4,
#                      1e-3, 5e-3]
weight_decay_list = [2e-7, 3e-7, 4e-7]
options = list(itertools.product(protein_names, flag_rmsd, weight_decay_list))

name, include_rmsd, weight_decay = options[rank]
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
fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

## read topology and pdb file
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)
xyz_init = traj_ref.xyz[0]

## minimize initial positions
platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)
context.setPositions(xyz_init)

state = context.getState(getEnergy = True)
potential_energy = state.getPotentialEnergy()
print(potential_energy)

omm.LocalEnergyMinimizer_minimize(context)
state = context.getState(getEnergy = True,
                         getPositions = True)
potential_energy = state.getPotentialEnergy()
print(potential_energy)
init_positions = state.getPositions()

os.makedirs(f"./output/{name}/NVT/", exist_ok = True)
file_handle = open(f"./output/{name}/NVT/rmsd_{include_rmsd}_weight_decay_{weight_decay:.3E}.dcd", 'wb')
dcd_file = app.DCDFile(file_handle, psf.to_openmm(), dt = 200*unit.femtoseconds)

start_time = time.time()
num_frames = 1_000_000
for i in range(num_frames):
    integrator.step(100)
    state = context.getState(getPositions = True)
    pos = state.getPositions()
    dcd_file.writeModel(pos)

    if (i + 1) % 100 == 0:
        print(i, flush = True)

print("time used: {:.2f}".format(time.time() - start_time))
file_handle.close()




