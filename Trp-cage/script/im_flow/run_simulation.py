#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=simulation_im_flow
#SBATCH --time=2-00:00:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
##SBATCH --array=0-11
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/simulation_im_flow.txt

import numpy as np
import openmm as omm
import openmm.unit as unit
import openmm.app as app
import math
import mdtraj
import pickle
from sys import exit
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG/")
from CLCG.utils.make_system import *
import argparse
import pandas as pd
import os
import torch
torch.set_default_dtype(torch.float64)
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import transform, MMFlow, utils
import torch.distributions as distributions
from openmmtorch import TorchForce
import time
from copy import copy

protein_info = pd.read_csv('./script/md/protein_temperature_and_ionic_strength.txt', comment = "#", index_col = 0)
protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 1
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
with open(f"./output/{name}/im_flow/system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

T = protein_info.loc[name, 'temperature']
fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)

traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)
xyz_init = traj_ref.xyz[0]
context.setPositions(xyz_init)

os.makedirs(f"./output/{name}/im_flow/CG_simulations/", exist_ok = True)
file_handle = open(f"./output/{name}/im_flow/CG_simulations/traj.dcd", 'wb')
dcd_file = app.DCDFile(file_handle, psf.to_openmm(), dt = 200*unit.femtoseconds)

start_time = time.time()
num_frames = 16000
for i in range(num_frames):
    integrator.step(100)
    state = context.getState(getPositions = True)
    pos = state.getPositions()
    dcd_file.writeModel(pos)

    if (i + 1) % 1 == 0:
        print(i, flush = True)

print("time used: {:.2f}".format(time.time() - start_time))
file_handle.close()

exit()
