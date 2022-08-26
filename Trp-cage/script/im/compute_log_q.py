#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=logq_im
#SBATCH --time=2:00:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-11
#SBATCH --mem=60G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/logq_im_%a.txt

import numpy as np
import simtk.openmm.app  as app
import simtk.openmm as omm
import simtk.unit as unit
import argparse
import mdtraj
from sys import exit
import time
import pickle
import pandas as pd
import os

protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
name = protein_names.iloc[job_idx, 0]

with open(f"./output/{name}/im/FF/CG_system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

protein_info = pd.read_csv("./script/md/protein_temperature_and_ionic_strength.txt", index_col = 'name', comment = '#')

## make an integrator
T = protein_info.loc[name, 'temperature']
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.kelvin * unit.AVOGADRO_CONSTANT_NA

fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)
platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)

## read topology and pdb file
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

traj_md = mdtraj.load_dcd(f"./data/traj_CG_stride_10/{name}.dcd", psf, stride = 1)
xyz_md = traj_md.xyz
log_q_md = []
for i in range(xyz_md.shape[0]):
    context.setPositions(xyz_md[i])
    state = context.getState(getEnergy = True)
    U = state.getPotentialEnergy()
    log_q_md.append(-U/kbT)

    if (i+1) % 10000 == 0:
        print(i)

traj_im = mdtraj.load_dcd(f"./output/{name}/im/CG_simulations/traj.dcd", psf, stride = 1)
xyz_im = traj_im.xyz
log_q_im = []
for i in range(xyz_im.shape[0]):
    context.setPositions(xyz_im[i])
    state = context.getState(getEnergy = True)
    U = state.getPotentialEnergy()
    log_q_im.append(-U/kbT)

    if (i+1) % 10000 == 0:
        print(i)
        
log_q_md = np.array(log_q_md)
log_q_im = np.array(log_q_im)

with open(f"./output/{name}/im/log_q.pkl", 'wb') as file_handle:
    pickle.dump({'log_q_md': log_q_md,
                 'log_q_im': log_q_im},
                file_handle)
exit()

