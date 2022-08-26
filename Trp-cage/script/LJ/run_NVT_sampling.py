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

## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
args = parser.parse_args()

name = args.name

weight_decay_list = [1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6]
job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
weight_decay = weight_decay_list[job_idx]

with open(f"./output/{name}/LJ/system/weight_decay_{weight_decay:.3E}.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

temperature_and_ionic_strength = pd.read_csv("./script/md/protein_temperature_and_ionic_strength.txt", index_col = 'name', comment = '#')

## make an integrator
T = temperature_and_ionic_strength.loc[name, 'temperature']
fricCoef = 1./unit.picoseconds
stepsize = 2. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

## read topology and pdb file
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)
xyz_init = traj_ref.xyz[0]

## minimize initial positions
platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)
context.setPositions(xyz_init)

os.makedirs(f"./output/{name}/LJ/CG_simulations/", exist_ok = True)
file_handle = open(f"./output/{name}/LJ/CG_simulations/weight_decay_{weight_decay:.3E}.dcd", 'wb')
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



