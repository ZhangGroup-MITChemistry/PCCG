#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=basis_LJ
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-30
#SBATCH --output=./slurm_output/basis_LJ_%a.txt
#SBATCH --open-mode=truncate

import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad
import pickle
import mdtraj
from sys import exit
import os
import argparse
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
from CLCG.utils.splines import *
from CLCG.utils.CL import *
import torch
torch.set_default_dtype(torch.double)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
parser.add_argument("--llsub_rank", type = int)
parser.add_argument("--llsub_size", type = int)
args = parser.parse_args()

name = args.name
llsub_rank = args.llsub_rank
llsub_size = args.llsub_size

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf, stride = 1)

size = 48
traj_im_us = []
for rank in range(size):
    traj = mdtraj.load_dcd(f"./output/{name}/im_us/trajs/traj_{rank}.dcd", psf, stride = 1)
    traj_im_us.append(traj)
    print(rank)
traj_im_us = mdtraj.join(traj_im_us)    

indices = [(i,j) for i in range(psf.n_residues) for j in range(i+4, psf.n_residues)]
num_of_basis = 8

if llsub_rank >= len(indices):
    exit()

pair = indices[llsub_rank]
r_md = mdtraj.compute_distances(traj_md, [pair])
r_md = np.squeeze(r_md)
r_min = r_md.min()
r_max = 1.5

basis_md, omega = bs_lj(r_md, r_min, r_max, num_of_basis, omega = True)
basis_info= {'r_min': r_min, 'r_max': r_max,
             'num_of_basis': num_of_basis, 'omega': omega}

r_im_us = mdtraj.compute_distances(traj_im_us, [pair])
r_im_us = np.squeeze(r_im_us)    
basis_im_us = bs_lj(r_im_us, r_min, r_max, num_of_basis, omega = False)    

r_over_range = np.linspace(0.2, r_max, 1000)
basis_over_range = bs_lj(r_over_range, r_min, r_max, num_of_basis, omega = False)

os.makedirs(f"./output/{name}/LJ/LJ_basis", exist_ok = True)
i,j = pair
with open(f"./output/{name}/LJ/LJ_basis/basis_info_{i}-{j}.pkl", 'wb') as file_handle:
    pickle.dump(basis_info, file_handle)
with open(f"./output/{name}/LJ/LJ_basis/basis_md_{i}-{j}.pkl", 'wb') as file_handle:
    pickle.dump(basis_md, file_handle)
with open(f"./output/{name}/LJ/LJ_basis/basis_im_us_{i}-{j}.pkl", 'wb') as file_handle:
    pickle.dump(basis_im_us, file_handle)
with open(f"./output/{name}/LJ/LJ_basis/basis_over_range_{i}-{j}.pkl", 'wb') as file_handle:
    pickle.dump({'r_over_range': r_over_range,
                 'basis_over_range':basis_over_range}, file_handle)

exit()

r_noise = np.random.uniform(0, r_md.max(), basis_md.shape[0])
basis_noise = bs_lj(r_noise, r_min, r_max, num_of_basis, omega = False)

log_q_noise = torch.log(torch.ones(len(r_noise))/r_md.max())
log_q_md = torch.log(torch.ones(basis_md.shape[0])/r_md.max())

alphas, F = contrastive_learning(log_q_noise, log_q_md, torch.from_numpy(basis_noise), torch.from_numpy(basis_md))

r_over_range = np.linspace(r_min - 0.1, 3.0, 10000)
basis_over_range = bs_lj(r_over_range, r_min, r_max, num_of_basis, omega = False)
u_p_over_range = np.matmul(basis_over_range, alphas)

fig = plt.figure(0, figsize = (6.4*2, 4.8))
fig.clf()
plt.subplot(1,2,1)
plt.hist(r_md, bins = 100, density = True, log = True, range = [0, 3.0])
plt.subplot(1,2,2)
plt.plot(r_over_range, u_p_over_range, label = 'CL')
plt.xlim(0, 3.0)
plt.ylim(u_p_over_range.min(), 15)
plt.legend()
plt.savefig(f"./output/{name}/LJ/LJ_basis/hist_r_and_learned_u_{pair[0]}-{pair[1]}.pdf")

    
exit()
