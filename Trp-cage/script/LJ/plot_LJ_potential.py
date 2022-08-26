#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=plot_LJ
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-8
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/plot_LJ_%a.txt

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from scipy.interpolate import BSpline
from sys import exit
import argparse
import mdtraj
import os

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
args = parser.parse_args()

name = args.name

weight_decay_list = [1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6]
job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
weight_decay = weight_decay_list[job_idx]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
resnames = [residue.name for residue in psf.residues]
for i in range(len(resnames)):
    if resnames[i] == "NLE":
        resnames[i] = "ILE"

## load spline coefficients
with open(f"./output/{name}/LJ/FF/weight_decay_{weight_decay:.3E}.pkl", 'rb') as file_handle:
    FF = pickle.load(file_handle)
    LJ_parameters = FF['LJ_parameters']
    rmsd_parameters = FF['rmsd_parameters']
    
pair_indices = list(LJ_parameters.keys())
pair_indices.sort()

ncol = 6
nrow = len(pair_indices)//ncol + 1

fig = plt.figure(np.random.randint(1000), figsize = (6.4*ncol, 4.8*nrow))
fig.clf()
for k in range(len(pair_indices)):
    plt.subplot(nrow,ncol,k+1)
    pair = pair_indices[k]
    i,j = pair_indices[k]

    aa1, aa2 = resnames[i], resnames[j]
    if resnames[j] < resnames[i]:
        aa1, aa2 = resnames[j], resnames[i]
    
    r_min, r_max = LJ_parameters[pair]['r_min'], LJ_parameters[pair]['r_max']
    ulj = LJ_parameters[pair]['U']
    r = LJ_parameters[pair]['r_over_range']
    plt.plot(r, ulj, linewidth = 6.0)
    plt.xlim(0, r_max)
    plt.ylim(ulj.min(), 10)
    plt.title(f"{i}-{j}:{aa1}-{aa2}")
    print(k)
    
fig.suptitle(f"weight_decay: {weight_decay:.3E}", fontsize = 42)
fig.savefig(f"./output/{name}/LJ/FF/LJ_weight_decay_{weight_decay:.3E}.pdf")

fig = plt.figure(np.random.randint(10000))
fig.clf()
plt.plot(rmsd_parameters['rmsd_over_range'], rmsd_parameters['U'])
fig.savefig(f"./output/{name}/LJ/FF/rmsd_weight_decay_{weight_decay:.3E}.pdf")
