#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=rmsd_imus
#SBATCH --time=00:20:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-1
#SBATCH --mem=60G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/rmsd_imus_%a.txt

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
import matplotlib.pyplot as plt
import pickle

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 1
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)

with open(f"./output/{name}/rmsd_centers_and_k_for_imus.pkl", 'rb') as file_handle:
    rmsd_centers_and_k = pickle.load(file_handle)
    
size = rmsd_centers_and_k['size']
rmsd = {}
for rank in range(size):
    print(f"rank: {rank}")
    traj_imus = mdtraj.load_dcd(f"./output/{name}/traj_imus/traj_{rank}.dcd", psf)
    rmsd[rank] = mdtraj.rmsd(traj_imus, traj_ref)

fig = plt.figure(figsize = (6.4*4, 4.8))
fig.clf()
for rank in range(size):
    plt.hist(rmsd[rank], 30, density = True, alpha = 0.4)
    #plt.axvline(rmsd_centers_and_k['centers'][rank], linestyle = '--')
plt.xlabel('RMSD (nm)')
plt.ylabel('probability density')
plt.savefig(f"./output/{name}/rmsd_hist_imus.pdf")
