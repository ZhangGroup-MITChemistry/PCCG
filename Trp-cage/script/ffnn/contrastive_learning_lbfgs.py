#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=ffnn
#SBATCH --time=10:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-13
#SBATCH --mem=100G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/ffnn_%a.txt

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
sys.path.append("./script/ffnn")
from NNForce import *
import os
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
args = parser.parse_args()

name = args.name

weight_decay_list = [1e-9, 5e-9,
                     1e-8, 5e-8,
                     1e-7, 5e-7,
                     1e-6, 5e-6,
                     1e-5, 5e-5,
                     1e-4, 5e-4,
                     1e-3, 5e-3]

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
weight_decay = weight_decay_list[job_idx]

print(f"weight_decay: {weight_decay:.3E}", flush = True)

protein_info = pd.read_csv('./script/md/protein_temperature_and_ionic_strength.txt', comment = "#", index_col = 0)
T = protein_info.loc[name, 'temperature']
T = T * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA
kbT_in_kJ_per_mole = kbT.value_in_unit(unit.kilojoule_per_mole)
beta = 1/kbT_in_kJ_per_mole

data = torch.load(f"./output/{name}/im/ic.pt")
ic_im = data['ic']

data = torch.load(f"./output/{name}/md/ic.pt")
ic_md = data['ic']

with open(f"./output/{name}/ffnn/helix_index.pkl", 'rb') as file_handle:
    helix_index = pickle.load(file_handle)

angle_md = ic_md.angle[:, helix_index['helix_angle_index']]
dihedral_md = ic_md.dihedral[:, helix_index['helix_dihedral_index']]

angle_im = ic_im.angle[:, helix_index['helix_angle_index']]
dihedral_im = ic_im.dihedral[:, helix_index['helix_dihedral_index']]

if helix_index['helix_reference_particle_3_angle_flag']:
    angle_md = torch.cat([ic_md.reference_particle_3_angle[:,None],
                          angle_md], dim = -1)
    angle_im = torch.cat([ic_im.reference_particle_3_angle[:,None],
                          angle_im], dim = -1)

angle = torch.cat([angle_im, angle_md])
dihedral = torch.cat([dihedral_im, dihedral_md])
target = torch.cat(
    [angle_im.new_zeros(angle_im.shape[0]),
     angle_md.new_ones(angle_md.shape[0])]
)

## nnforce on angles and dihedrals
hidden_size = 32
num_angles = angle.shape[-1]
num_dihedrals = dihedral.shape[-1]
nnforce_angle_dihedral = NNForceAngleDihedral(
    num_angles,
    num_dihedrals,
    hidden_size
)

with torch.no_grad():
    energy_p = nnforce_angle_dihedral(angle, dihedral)
energy_q = torch.zeros_like(energy_p)

energy_matrix = np.vstack([energy_q, energy_p])
num_conf = np.array([angle_im.shape[0], angle_md.shape[0]])
fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True)
F = fastmbar.F[-1]

## contrastive learning
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
nnforce_angle_dihedral = nnforce_angle_dihedral.to(device)
angle = angle.to(device)
dihedral = dihedral.to(device)
target = target.to(device)

F = torch.tensor([F], device = device, requires_grad = True)

num_eval = [0]
params = list(nnforce_angle_dihedral.parameters()) + [F]
#numel = reduce(lambda total, p: total + p.numel(), params, 0)

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
    nnforce_energy = nnforce_angle_dihedral(angle, dihedral)
    logit = -nnforce_energy
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)

    for p in nnforce_angle_dihedral.parameters():
        loss = loss + weight_decay*torch.sum(p**2)
    loss.backward()

    grad = gather_flat(params, grad = True)
    
    return loss.item(), grad.clone().cpu().numpy()

loss, grad = compute_loss_and_grad(x_init, weight_decay = 0.0)
x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 x_init,
                                 args = [weight_decay],
                                 iprint = 1)
set_param(params, torch.from_numpy(x))
final_loss, _ = compute_loss_and_grad(x, weight_decay = 0.0)
print(f"final_loss: {final_loss:.6E}")

os.makedirs(f"./output/{name}/ffnn/FF", exist_ok = True)    
torch.save({'num_angles': num_angles,
            'num_dihedrals': num_dihedrals,
            'hidden_size': hidden_size,
            'state_dict':nnforce_angle_dihedral.state_dict()},
           f"./output/{name}/ffnn/FF/NNForceAngleDihedral_weight_decay_{weight_decay:.3E}.pth")


            
