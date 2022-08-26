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
from itertools import product
import pickle

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
protein_names = protein_names.iloc[:, 0].tolist()

# weight_decay_list = [1e-8,
#                      1e-7, 2e-7, 5e-7,
#                      1e-6, 2e-6, 5e-6,
#                      1e-5]

weight_decay_list = [5e-7,
                     1e-6, 2e-6]

#elec_type_list = ['simple', 'fshift']
elec_type_list = ['DH_2']

#ss_type_list = ['simple', 'extended']
ss_type_list = ['simple']

options = list(product(protein_names,
                       elec_type_list,
                       ss_type_list,
                       weight_decay_list))

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
#job_idx = rank
if rank >= len(options):
    exit()

name, elec_type, ss_type, weight_decay = options[rank]

print(f"name: {name}, elec_type: {elec_type}, ss_type: {ss_type}, weight_decay: {weight_decay}", flush = True)

## read system
with open(f"./output/{name}/full_system/elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.xml", 'r') as file_handle:
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

os.makedirs(f"./output/{name}/NVT_long/", exist_ok = True)

num_chunk = 10
for idx_chunk in range(num_chunk):
    file_handle = open(f"./output/{name}/NVT_long/elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}_chunk_{idx_chunk}.dcd", 'wb')
    dcd_file = app.DCDFile(file_handle, psf.to_openmm(), dt = 200*unit.femtoseconds)

    start_time = time.time()
    num_frames = 1_000_000
    
    for i in range(num_frames):
        integrator.step(100)
        state = context.getState(getPositions = True)
        pos = state.getPositions()
        dcd_file.writeModel(pos)

        if (i + 1) % 1000 == 0:
            print(f"idx_chunk: {idx_chunk}, frame: {i}", flush = True)

    print("time used: {:.2f}".format(time.time() - start_time))
    file_handle.close()

    checkpoint = context.createCheckpoint()
    with open(f"./output/{name}/NVT_long/elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}_chunk_{idx_chunk}.checkpoint", 'wb') as file_handle:
        pickle.dump(checkpoint, file_handle)


    
    
    
