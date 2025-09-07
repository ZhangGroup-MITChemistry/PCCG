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
parser.add_argument('--rank', type = int)
parser.add_argument('--size', type = int)
args = parser.parse_args()

rank = args.rank
size = args.size
print(f"rank: {rank}, size: {size}")

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
protein_info = pd.read_csv("./info/protein_temperature_and_ionic_strength.txt",
                           index_col = 'name', comment = '#')
protein_names = protein_names.iloc[:, 0].tolist()
rmsd_centers_and_k = {}
for name in protein_names:
    with open(f"./output/{name}/rmsd_centers_and_k_for_imus.pkl", 'rb') as file_handle:
        data = pickle.load(file_handle)
        rmsd_centers_and_k[name] = data
sizes = [rmsd_centers_and_k[name]['size'] for name in protein_names]
cumsum_sizes = np.cumsum(sizes)

exit()

if rank >= cumsum_sizes[-1]:
    exit()

idx_name = np.searchsorted(cumsum_sizes, rank, side = 'right')
name = protein_names[idx_name]

if idx_name == 0:
    idx_window = rank
else:
    idx_window = rank - cumsum_sizes[idx_name - 1]    

rmsd_center = rmsd_centers_and_k[name]['centers'][idx_window]
rmsd_k = rmsd_centers_and_k[name]['k']

print("sizes:", sizes)        
print("cumsum_sizes:", cumsum_sizes)
print(f"rank: {rank}, name: {name}, idx_window: {idx_window}")
print(f"rmsd_k: {rmsd_k:.2f}, rmsd_center: {rmsd_center:.3f}")

with open(f"./output/{name}/system_im.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)
xyz_init = traj_ref.xyz[0]

particle_index = list(range(psf.n_residues))
rmsd_force = omm.RMSDForce(traj_ref.xyz[0], particle_index)
cv_force = omm.CustomCVForce("0.5*rmsd_k*(rmsd - rmsd_0)^2")
cv_force.addCollectiveVariable('rmsd', rmsd_force)
cv_force.addGlobalParameter('rmsd_k', rmsd_k)
cv_force.addGlobalParameter('rmsd_0', rmsd_center)
system.addForce(cv_force)

## make an integrator
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
    print(k)
    
context.setParameter('rmsd_k', rmsd_k)
integrator.step(10000)

os.makedirs(f"./output/{name}/traj_imus", exist_ok = True)
file_handle = open(f"./output/{name}/traj_imus/traj_{idx_window}.dcd", 'wb')
dcd_file = app.DCDFile(file_handle, psf.to_openmm(), dt = 200*unit.femtoseconds)

traj_md = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", psf)

start_time = time.time()
num_frames = 25_000

print(f"num_frames: {num_frames}")

for i in range(num_frames):
    integrator.step(1000)
    state = context.getState(getPositions = True)
    pos = state.getPositions()
    dcd_file.writeModel(pos)

    if (i + 1) % 100 == 0:
        print(i, flush = True)

print("time used: {:.2f}".format(time.time() - start_time))
file_handle.close()        
