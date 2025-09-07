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

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 10
name = protein_names.iloc[job_idx, 0]

print(f"protein: {name}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

ref_particle_1 = psf.n_atoms//2
ref_particle_2 = ref_particle_1 - 1
ref_particle_3 = ref_particle_1 + 1

bonds = {}
for i in range(psf.n_atoms):
    bonds[i] = []
    if i - 1 >= 0:
        bonds[i].append(i-1)
    if i + 1 < psf.n_atoms:
        bonds[i].append(i+1)

coor_transformer = utils.CoordinateTransformer(
    bonds,
    ref_particle_1,
    ref_particle_2,
    ref_particle_3,
    dihedral_mode = 'linear'
)

os.makedirs(f"./output/{name}/", exist_ok = True)
with open(f"./output/{name}/coor_transformer.pkl", 'wb') as file_handle:
    pickle.dump(coor_transformer, file_handle)

traj = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", psf)

xyz = traj.xyz.astype(np.float64)
ic, ic_logabsdet = coor_transformer.compute_internal_coordinate_from_xyz(torch.from_numpy(xyz))

exit()

os.makedirs(f"./output/{name}/", exist_ok = True)
torch.save({'ic': ic, 'ic_logabsdet': ic_logabsdet},
           f"./output/{name}/ic_md.pt")

ic.plot(file_name = f"./output/{name}/ic_md_hist.pdf")
