#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=bad_dist
#SBATCH --time=00:10:00
#SBATCH --partition=xeon-p8
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --array=0-1
#SBATCH --mem=30G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/bad_dist_%a.txt

import mdtraj
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import numpy as np
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
import pickle
from sys import exit
import pandas as pd
import os
import torch

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
with open(f"./output/{name}/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)
    
data = torch.load(f"./output/{name}/ic_md.pt")
ic_md = data['ic']

traj_im = mdtraj.load_dcd(f"./output/{name}/traj_im/traj.dcd", psf)
xyz = torch.from_numpy(traj_im.xyz)
ic_im, _ = coor_transformer.compute_internal_coordinate_from_xyz(xyz)

bond_md = torch.cat([ic_md.reference_particle_2_bond[:, None],
                     ic_md.reference_particle_3_bond[:, None],
                     ic_md.bond], dim = -1).numpy()

bond_im = torch.cat([ic_im.reference_particle_2_bond[:, None],
                     ic_im.reference_particle_3_bond[:, None],
                     ic_im.bond], dim = -1).numpy()

angle_md = torch.cat([ic_md.reference_particle_3_angle[:, None],
                      ic_md.angle], dim = -1).numpy()

angle_im = torch.cat([ic_im.reference_particle_3_angle[:, None],
                      ic_im.angle], dim = -1).numpy()

dihedral_md = ic_md.dihedral.numpy()
dihedral_im = ic_im.dihedral.numpy()

num_columns = 4
num_rows = (ic_im.angle.shape[-1]*3 + 3) // num_columns + 1
fig = plt.figure(np.random.randint(1000), figsize = (6.4*num_columns, 4.8*num_rows))
fig.clf()

subplot_idx = 1
stride = 10

for j in range(bond_md.shape[-1]):
    plt.subplot(num_rows, num_columns, subplot_idx)    
    plt.hist(bond_md[::stride,j], bins = 30, density = True, label = f"bond_md", alpha = .5)
    plt.hist(bond_im[::stride,j], bins = 30, density = True, label = f"bond_im", alpha = .5)
    plt.title('bond')
    plt.legend()
    subplot_idx += 1
    print(subplot_idx)
    
for j in range(angle_md.shape[-1]):
    plt.subplot(num_rows, num_columns, subplot_idx)    
    plt.hist(angle_md[::stride,j], bins = 30, density = True, label = f"angle_md", alpha = .5)
    plt.hist(angle_im[::stride,j], bins = 30, density = True, label = f"angle_im", alpha = .5)
    plt.title('angle')
    plt.legend()
    subplot_idx += 1
    print(subplot_idx)    
    
for j in range(dihedral_md.shape[-1]):
    plt.subplot(num_rows, num_columns, subplot_idx)    
    plt.hist(dihedral_md[::stride,j], bins = 30, density = True, label = f"dihedral_md", alpha = .5)
    plt.hist(dihedral_im[::stride,j], bins = 30, density = True, label = f"dihedral_im", alpha = .5)
    plt.title('dihedral')
    plt.legend()
    subplot_idx += 1
    print(subplot_idx)    
    
plt.tight_layout()
plt.savefig(f"./output/{name}/bad_hist_im.pdf")

exit()
