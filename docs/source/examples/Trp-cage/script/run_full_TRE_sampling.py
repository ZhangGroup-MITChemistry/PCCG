#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=TRE
#SBATCH --time=3-00:00:00
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=25
#SBATCH --exclusive=user
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-27
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/TRE_%a.txt

import numpy as np
import openmm.app  as app
import openmm as omm
import openmm.unit as unit
import math
from sys import exit
import ray
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMRay")
from TRE import *
import argparse
import time
import pickle
import os
import pandas as pd
import mdtraj
import argparse

ray.init(num_cpus = int(os.environ['SLURM_CPUS_PER_TASK']),
         _temp_dir = '/home/gridsan/dingxq/tmp/ray')
print("available_resources: ", ray.available_resources())

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
weight_decay_list = [1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6]

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
protein_idx = job_idx // len(weight_decay_list)
weight_decay_idx = job_idx % len(weight_decay_list)

name = protein_names.iloc[protein_idx, 0]
weight_decay = weight_decay_list[weight_decay_idx]

print(f"name: {name}, weight_decay: {weight_decay}", flush = True)


## read system
with open(f"./output/{name}/full_system/weight_decay_{weight_decay:.3E}.xml", 'r') as file_handle:
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

#### running TREMD
## construct TREActor
low_T = protein_info.loc[name, 'temperature']
high_T = 600
num_temperatures = int(os.environ['SLURM_CPUS_PER_TASK'])

temperatures =  1./np.linspace(1./low_T, 1./high_T, num_temperatures)
os.makedirs(f"./output/{name}/TRE/weight_decay_{weight_decay:.3E}", exist_ok = True)
dcd_file_names = [f"./output/{name}/TRE/weight_decay_{weight_decay:.3E}/traj_{i}.dcd"
                  for i in range(num_temperatures)]
actors = \
[ TREActor.options(num_cpus=1).remote(
    system, integrator, "Reference",
    temperatures[i], psf.to_openmm(),
    dcd_file_names[i], init_positions)
  for i in range(num_temperatures)]

## set all initial temperature to 100K
initial_T = 100.
[actor.update_temperature.remote(initial_T) for actor in actors]

## slowly increase temperature to the target tempearture for each replica
num_heating_steps = 125
dT = [(T - initial_T)/num_heating_steps for T in temperatures]
for idx_heating_step in range(num_heating_steps):
    print("idx_heating_step: {}".format(idx_heating_step))
    [actor.run_md.remote(1000) for actor in actors]
    [actors[i].update_temperature.remote(initial_T + (idx_heating_step + 1)*dT[i]) for i in range(len(actors))]
    
[actor.run_md.remote(10000) for actor in actors]

print("target temperatures:", temperatures)
print("final tempeartures:", ray.get([actor.get_temperature.remote() for actor in actors]))

start_time = time.time()

## run TREMD
num_steps = int(1e8)
exchange_freq = int(1e3)
save_freq = int(1e3)

# num_steps = int(2e5)
# exchange_freq = int(1e3)
# save_freq = int(1e3)

tre = TRE(actors)
tre.run(num_steps, exchange_freq, save_freq)

log = ray.get([actors[i].get_temperature_record.remote() for i in range(num_temperatures)])
log = np.array(log)

with open(f"./output/{name}/TRE/weight_decay_{weight_decay:.3E}/log_temperatures.pkl", 'wb') as file_handle:
    pickle.dump(log, file_handle)

print("used time: {:.2f}".format(time.time() - start_time))
ray.shutdown()    

exit()
