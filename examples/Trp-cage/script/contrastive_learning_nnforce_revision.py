#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=nnforce
#SBATCH --time=10:00:00
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --mem=100G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/nnforce_%a.txt

import argparse
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import pickle
from scipy import optimize
from sys import exit
import mdtraj
import pandas as pd
import simtk.unit as unit
from FastMBAR import *
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
sys.path.append("./script")
from NNForce import *
import os
from functools import reduce

name = '2JOF'
weight_decay_list = [1e-3]

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 0
weight_decay = weight_decay_list[job_idx]

print(f"weight_decay: {weight_decay:.3E}", flush = True)

full_include_rmsd = False
full_weight_decay = 4e-7

print(f"full rmsd: {full_include_rmsd}, weight_decay: {full_weight_decay:.3E}")

with open(f"./output/{name}/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
traj_noise = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay:.3E}.dcd", psf)

ic_md, _ = coor_transformer.compute_internal_coordinate_from_xyz(torch.from_numpy(traj_md.xyz))
ic_noise, _ = coor_transformer.compute_internal_coordinate_from_xyz(torch.from_numpy(traj_noise.xyz))

angle_md = torch.cat([ic_md.reference_particle_3_angle[:, None], ic_md.angle], dim = 1).double()
angle_noise = torch.cat([ic_noise.reference_particle_3_angle[:, None], ic_noise.angle], dim = 1).double()
angle = torch.cat([angle_noise, angle_md])

dihedral_md = ic_md.dihedral.double()
dihedral_noise = ic_noise.dihedral.double()
dihedral = torch.cat([dihedral_noise, dihedral_md])

p_index_for_distances = torch.LongTensor([[i,j] for i in range(psf.n_atoms) for j in range(i+4, psf.n_atoms)])
distance_md = utils.functional.compute_distances(torch.from_numpy(traj_md.xyz),
                                                 p_index_for_distances)

distance_noise = utils.functional.compute_distances(torch.from_numpy(traj_noise.xyz),
                                                   p_index_for_distances)
distance = torch.cat([distance_noise, distance_md])

target = torch.cat([torch.zeros(len(ic_noise)), torch.ones(len(ic_md))])

## log_p
hidden_size = 32
torch.manual_seed(0)
nnforce_net = NNForceNet(angle.shape[-1],
                         dihedral.shape[-1],
                         distance.shape[-1],
                         hidden_size)

with torch.no_grad():
    nnforce_energy = nnforce_net(angle, dihedral, distance)
    
energy_md = nnforce_energy
energy_noise = torch.zeros_like(energy_md)

energy_matrix = torch.stack([energy_noise, energy_md]).numpy()
num_conf = np.array([traj_noise.n_frames, traj_md.n_frames])
fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True)
F = fastmbar.F[-1]

## contrastive learning
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

nnforce_net = nnforce_net.to(device)
F = torch.tensor([F], device = device, requires_grad = True)

angle = angle.to(device)
dihedral = dihedral.to(device)
distance = distance.to(device)
energy_noise = energy_noise.to(device)
log_q = -energy_noise
target = target.to(device)

num_eval = [0]

params = list(nnforce_net.parameters()) + [F]

def gather_flat(params, grad = False):
    views = []
    for p in params:
        if grad:
            if p.grad is None:
                view =  p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def set_param(params, update):
    update = update.to(device)
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(update[offset:offset + numel].view_as(p))
        offset += numel

x_init = gather_flat(params).detach().clone().cpu().numpy()
set_param(params, torch.from_numpy(x_init))

def compute_loss_and_grad(x, weight_decay):
    ## set all gradients to zero
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    ## set parameters for the neural network
    set_param(params, torch.from_numpy(x))

    ## compute loss and gradient
    nnforce_energy = nnforce_net(angle, dihedral, distance)
    energy = nnforce_energy
    log_p = F - energy
    
    logit = log_p - log_q
    loss = nn.functional.binary_cross_entropy_with_logits(
        logit, target, reduction = 'mean')
    for p in nnforce_net.parameters():
        loss = loss + weight_decay*torch.sum(p**2)
        
    loss.backward()
    loss = loss.cpu().item()
    grad = gather_flat(params, grad = True)
    return loss, grad.cpu().numpy()

loss, grad = compute_loss_and_grad(x_init, weight_decay = 0.0)
x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 x_init,
                                 args = [weight_decay],
                                 iprint = 1,
                                 maxfun = 50000,
                                 maxiter = 50000)

set_param(params, torch.from_numpy(x))
loss_wo_penalty, _ = compute_loss_and_grad(x, weight_decay = 0.0)

torch.save({'hidden_size': hidden_size,
            'state_dict':nnforce_net.cpu().state_dict(),
            'p_index_for_distances': p_index_for_distances,                        
            'loss_wo_penalty': loss_wo_penalty,},
           f"./output/{name}/FF/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_NNForce_weight_decay_{weight_decay:.3E}_different_initial_parameters.pth")
