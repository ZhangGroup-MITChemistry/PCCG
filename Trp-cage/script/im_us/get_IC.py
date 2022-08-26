#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=get_IC
#SBATCH --time=00:10:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-11
#SBATCH --open-mode=truncate
#SBATCH --mem=30G
#SBATCH --output=./slurm_output/get_IC_%a.txt

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import argparse
import os
import mdtraj
from sys import exit
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
import torch
torch.set_default_dtype(torch.float)
import pickle
import pandas as pd

protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 1
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

with open(f"./output/{name}/im_us/rmsd_centers_and_k.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
rmsd_centers = data['centers']
rmsd_k = data['k']

size = len(rmsd_centers)
trajs = []
for rank in range(size):
    traj = mdtraj.load_dcd(f"./output/{name}/im_us/trajs/traj_{rank}.dcd", psf, stride = 1)
    trajs.append(traj)
    print(rank)
traj = mdtraj.join(trajs)    
xyz = traj.xyz.astype(np.float64)
ic, ic_logabsdet = coor_transformer.compute_internal_coordinate_from_xyz(torch.from_numpy(xyz))

os.makedirs(f"./output/{name}/im_us/", exist_ok = True)
torch.save({'ic': ic, 'ic_logabsdet': ic_logabsdet},
           f"./output/{name}/im_us/ic.pt")
