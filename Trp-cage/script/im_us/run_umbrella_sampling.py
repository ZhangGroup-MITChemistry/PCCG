#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=NVT_IM_US
#SBATCH --time=03:00:00
#SBATCH --partition=xeon-p8
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=2
##SBATCH --nodes=1
##SBATCH --array=0-39
##SBATCH --open-mode=truncate
##SBATCH --output=./slurm_output/NVT_IM_US_%a.txt


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
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
parser.add_argument('--rank', type = int)
parser.add_argument('--size', type = int)
args = parser.parse_args()

name = args.name
rank = args.rank
size = args.size
print(f"name: {name}, rank: {rank}, size: {size}")

with open(f"./output/{name}/im/FF/CG_system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)
xyz_init = traj_ref.xyz[0]

with open(f"./output/{name}/im_us/rmsd_centers_and_k.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
rmsd_centers = data['centers']
rmsd_k = data['k']
assert(size == len(rmsd_centers))

particle_index = list(range(psf.n_residues))
rmsd_force = omm.RMSDForce(traj_ref.xyz[0], particle_index)
cv_force = omm.CustomCVForce("0.5*rmsd_k*(rmsd - rmsd_0)^2")
cv_force.addCollectiveVariable('rmsd', rmsd_force)
cv_force.addGlobalParameter('rmsd_k', rmsd_k)
cv_force.addGlobalParameter('rmsd_0', rmsd_centers[rank])
system.addForce(cv_force)

print(f"rmsd_k: {rmsd_k:.3f}, rmsd_0: {rmsd_centers[rank]:.3f}")

## make an integrator
protein_info = pd.read_csv("./script/md/protein_temperature_and_ionic_strength.txt", index_col = 'name', comment = '#')
T = protein_info.loc[name, 'temperature']
fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)
context.setPositions(xyz_init)

for k in np.linspace(0, rmsd_k, 1000):
    context.setParameter('rmsd_k', k)
    integrator.step(1000)    
    
context.setParameter('rmsd_k', rmsd_k)
integrator.step(10000)

os.makedirs(f"./output/{name}/im_us/trajs/", exist_ok = True)
file_handle = open(f"./output/{name}/im_us/trajs/traj_{rank}.dcd", 'wb')
dcd_file = app.DCDFile(file_handle, psf.to_openmm(), dt = 200*unit.femtoseconds)

start_time = time.time()
num_frames = 40_000
for i in range(num_frames):
    integrator.step(1000)
    state = context.getState(getPositions = True)
    pos = state.getPositions()
    dcd_file.writeModel(pos)

    if (i + 1) % 100 == 0:
        print(i, flush = True)

print("time used: {:.2f}".format(time.time() - start_time))
file_handle.close()        
