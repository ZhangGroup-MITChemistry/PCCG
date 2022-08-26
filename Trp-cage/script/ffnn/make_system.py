#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=make_system_ffnn
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-13
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/make_system_ffnn_%a.txt

import simtk.openmm.app  as app
import simtk.openmm as omm
import simtk.unit as unit
import pickle
import math
from sys import exit
from collections import defaultdict
import numpy as np
import mdtraj
from collections import defaultdict
import pandas as pd
import argparse
from scipy.interpolate import BSpline
import os
from openmmtorch import TorchForce
import copy
import time
import torch
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import transform, MMFlow, utils
sys.path.append("./script/ffnn")
from NNForce import *

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
args = parser.parse_args()

name = args.name
weight_decay_list = [1e-9, 5e-9,
                     1e-8, 5e-8,
                     1e-7, 5e-7,
                     1e-6, 5e-6,
                     1e-5, 5e-5,
                     1e-4, 5e-4,
                     1e-3, 5e-3]

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
weight_decay = weight_decay_list[job_idx]

with open(f"./output/{name}/im/FF/CG_system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

data = torch.load(f"./output/{name}/ffnn/FF/NNForceAngleDihedral_weight_decay_{weight_decay:.3E}.pth",
                  map_location = torch.device('cpu'))
num_angles = data['num_angles']
num_dihedrals = data['num_dihedrals']
hidden_size = data['hidden_size']
nnforce_angle_dihedral_state_dict = data['state_dict']

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

with open(f"./output/{name}/ffnn/helix_index.pkl", 'rb') as file_handle:
    helix_index = pickle.load(file_handle)

particle_index_for_angle = helix_index['helix_angle_particle_index']
if helix_index['helix_reference_particle_3_angle_flag']:
    particle_index_for_angle += [[coor_transformer.ref_particle_2,
                                  coor_transformer.ref_particle_1,
                                  coor_transformer.ref_particle_3]]
particle_index_for_dihedral = helix_index['helix_dihedral_particle_index']    

protein_info = pd.read_csv('./script/md/protein_temperature_and_ionic_strength.txt', comment = "#", index_col = 0)
T = protein_info.loc[name, 'temperature']

nnforce_xyz = NNForceXYZ(torch.tensor(T),
                         hidden_size,
                         torch.tensor(particle_index_for_angle),
                         torch.tensor(particle_index_for_dihedral))
nnforce_xyz.nnforce_angle_dihedral.load_state_dict(nnforce_angle_dihedral_state_dict)

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)
xyz_init = traj_ref.xyz[0].astype(np.float64)

traced_nnforce_xyz = torch.jit.trace(nnforce_xyz, torch.from_numpy(xyz_init))
traced_nnforce_xyz.save(f"./output/{name}/ffnn/FF/nnforce_xyz_weight_decay_{weight_decay:.3E}.pth")

nnforce = TorchForce(f"./output/{name}/ffnn/FF/nnforce_xyz_weight_decay_{weight_decay:.3E}.pth")
nnforce.setForceGroup(0)
system.addForce(nnforce)

xml = omm.XmlSerializer.serialize(system)
os.makedirs(f"./output/{name}/ffnn/system", exist_ok = True)
with open(f"./output/{name}/ffnn/system/system_weight_decay_{weight_decay:.3E}.xml", 'w') as file_handle:
    file_handle.write(xml)

fricCoef = 1./unit.picoseconds
stepsize = 2. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)
context.setPositions(xyz_init)

state = context.getState(getEnergy = True)
potential_energy = state.getPotentialEnergy()
print(potential_energy)

context.getState(getEnergy=True)
start_time = time.time()
for i in range(30):
    integrator.step(100)
    context.getState(getEnergy=True)
    print(i)
used_time = time.time() - start_time
print(f"used time: {used_time:.2f}")    
