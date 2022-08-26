#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=CL_LJ
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-8
#SBATCH --mem=200G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/CL_LJ_%a.txt

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
import pyarrow.feather as feather
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
args = parser.parse_args()

name = args.name

weight_decay_list = [1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6]
job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
weight_decay = weight_decay_list[job_idx]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

## load internal coordinates from md and noise
data = torch.load(f"./output/{name}/md/ic.pt")
ic_md = data['ic']
ic_md.double()

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

data = torch.load(f"./output/{name}/im_us/ic.pt")
ic_noise = data['ic']
ic_noise.double()

with open(f"./output/{name}/im/FF/bonded_parameters.pkl", 'rb') as file_handle:
    bonded_parameters = pickle.load(file_handle)

b0 = np.array([bonded_parameters['reference_particle_2_bond']['b0'],
               bonded_parameters['reference_particle_3_bond']['b0']] + \
              list(bonded_parameters['bond']['b0']))
b0 = torch.from_numpy(b0)

kb = np.array([bonded_parameters['reference_particle_2_bond']['kb'],
               bonded_parameters['reference_particle_3_bond']['kb']] + \
              list(bonded_parameters['bond']['kb']))
kb = torch.from_numpy(kb)

bond_md = torch.cat(
    (ic_md.reference_particle_2_bond[:, None],
     ic_md.reference_particle_3_bond[:, None],
     ic_md.bond), dim = -1)

bond_noise = torch.cat(
    (ic_noise.reference_particle_2_bond[:, None],
     ic_noise.reference_particle_3_bond[:, None],
     ic_noise.bond), dim = -1)

ca = np.array([bonded_parameters['reference_particle_3_angle']['alphas']] +
              [p['alphas'] for p in bonded_parameters['angle']])
ca = torch.from_numpy(ca)

with open(f"./output/{name}/md/basis_angle_and_dihedrals.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    basis_md = data['basis']

with open(f"./output/{name}/im_us/basis_angle_and_dihedrals.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    basis_noise = data['basis']

with open(f"./output/{name}/md/basis_over_range_angle_and_dihedrals.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    basis_over_range = data['basis_over_range']
    angle_over_range = data['angle_over_range']
    dihedral_over_range = data['dihedral_over_range']
    
angle_md_basis = np.concatenate((basis_md['reference_particle_3_angle'][:,np.newaxis,:],
                                 basis_md['angle']), axis = 1)
angle_md_basis = torch.from_numpy(angle_md_basis)

angle_noise_basis = np.concatenate((basis_noise['reference_particle_3_angle'][:,np.newaxis,:],
                                    basis_noise['angle']), axis = 1)
angle_noise_basis = torch.from_numpy(angle_noise_basis)

angle_over_range_basis = basis_over_range['angle']

cd = np.array([p['alphas'] for p in bonded_parameters['dihedral']])
cd = torch.from_numpy(cd)

dihedral_md_basis = torch.from_numpy(basis_md['dihedral'])
dihedral_noise_basis = torch.from_numpy(basis_noise['dihedral'])
dihedral_over_range_basis = basis_over_range['dihedral']


## pair wise distances
pair_indices = [(i,j) for i in range(psf.n_residues) for j in range(i+4, psf.n_residues)]
LJ_md_basis = {}
for (i,j) in pair_indices:
    with open(f"./output/{name}/LJ/LJ_basis/basis_md_{i}-{j}.pkl", 'rb') as file_handle:
        LJ_md_basis[(i,j)] = pickle.load(file_handle)

LJ_noise_basis = {}
for (i,j) in pair_indices:
    with open(f"./output/{name}/LJ/LJ_basis/basis_im_us_{i}-{j}.pkl", 'rb') as file_handle:
        LJ_noise_basis[(i,j)] = pickle.load(file_handle)

LJ_over_range_basis = {}
r_over_range = {}
for (i,j) in pair_indices:
    with open(f"./output/{name}/LJ/LJ_basis/basis_over_range_{i}-{j}.pkl", 'rb') as file_handle:
        data = pickle.load(file_handle)
        LJ_over_range_basis[(i,j)] = data['basis_over_range']
        r_over_range[(i,j)] = data['r_over_range']

LJ_md_basis = np.stack([LJ_md_basis[k] for k in pair_indices], axis = 1)
LJ_md_basis = torch.from_numpy(LJ_md_basis)

LJ_noise_basis = np.stack([LJ_noise_basis[k] for k in pair_indices], axis = 1)
LJ_noise_basis = torch.from_numpy(LJ_noise_basis)

with open(f"./output/{name}/md/basis_rmsd.pkl", 'rb') as file_handle:
    rmsd_md_basis = pickle.load(file_handle)
    rmsd_md_basis = torch.from_numpy(rmsd_md_basis)
with open(f"./output/{name}/im_us/basis_rmsd.pkl", 'rb') as file_handle:
    rmsd_noise_basis = pickle.load(file_handle)
    rmsd_noise_basis = torch.from_numpy(rmsd_noise_basis)    
with open(f"./output/{name}/md/basis_rmsd_over_range.pkl", 'rb') as file_handle:    
    data = pickle.load(file_handle)
    rmsd_over_range_basis = data['basis_over_range']
    rmsd_over_range = data['rmsd_over_range']
    
## load log_q
with open(f"./output/{name}/im_us/log_q.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
log_q_md = data['log_q_md']
log_q_noise = data['log_q_us']

temperature_and_ionic_strength = pd.read_csv("./script/md/protein_temperature_and_ionic_strength.txt", index_col = 'name', comment = '#')
T = temperature_and_ionic_strength.loc[name, 'temperature']
T = T * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA
kbT_in_kJ_per_mole = kbT.value_in_unit(unit.kilojoule_per_mole)
beta = 1/kbT_in_kJ_per_mole

log_q = np.concatenate([log_q_noise, log_q_md])

## log_p
bond_energy_md = torch.sum(0.5*kb*(bond_md - b0)**2, -1)
angle_energy_md = torch.sum(ca*angle_md_basis, dim = (1,2))
dihedral_energy_md = torch.sum(cd*dihedral_md_basis, dim = (1,2))

bond_energy_noise = torch.sum(0.5*kb*(bond_noise - b0)**2, -1)
angle_energy_noise = torch.sum(ca*angle_noise_basis, dim = (1,2))
dihedral_energy_noise = torch.sum(cd*dihedral_noise_basis, dim = (1,2))

bonded_energy_md = bond_energy_md + angle_energy_md + dihedral_energy_md
bonded_energy_noise = bond_energy_noise + angle_energy_noise + dihedral_energy_noise

log_p_md = -bonded_energy_md
log_p_noise = -bonded_energy_noise

log_p = np.concatenate([log_p_noise.numpy(), log_p_md.numpy()])

energy_matrix = -np.vstack([log_q, log_p])
num_conf = np.array([len(log_q_noise), len(log_q_md)])

fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True)
F = np.array([fastmbar.F[-1]])

## contrastive learning
num_angle, num_basis_per_angle = ca.shape
num_dihedral, num_basis_per_dihedral = cd.shape
num_LJ, num_basis_per_LJ = LJ_md_basis.shape[1:]
num_basis_rmsd = rmsd_md_basis.shape[-1]

def flatten_parameters(ca, cd, clj, crmsd, F):
    c = torch.cat( (ca.reshape(-1), cd.reshape(-1), clj.reshape(-1), crmsd, F) )
    return c

def unflatten_parameters(c):
    idx_start = 0
    idx_end = idx_start + num_angle*num_basis_per_angle
    ca = c[idx_start:idx_end].reshape((num_angle, num_basis_per_angle))
    
    idx_start = idx_end
    idx_end = idx_start + num_dihedral*num_basis_per_dihedral
    cd = c[idx_start:idx_end].reshape((num_dihedral, num_basis_per_dihedral))

    idx_start = idx_end
    idx_end = idx_start + num_LJ*num_basis_per_LJ
    clj = c[idx_start:idx_end].reshape((num_LJ, num_basis_per_LJ))

    idx_start = idx_end
    idx_end = idx_start + num_basis_rmsd
    crmsd = c[idx_start:idx_end]
    
    F = c[-1]
    
    return ca, cd, clj, crmsd, F

ca_init = ca.clone()
cd_init = cd.clone()
clj_init = torch.zeros((num_LJ, num_basis_per_LJ))
crmsd_init = torch.zeros(num_basis_rmsd)
F_init = torch.from_numpy(F)
c_init = flatten_parameters(ca_init, cd_init, clj_init, crmsd_init, F_init)

## bounds for the optimization
bounds = []
for _ in range(ca_init.numel() + cd_init.numel()):
    bounds.append([None, None])

for i in range(clj_init.shape[0]):
    for j in range(clj_init.shape[1]):
        if j == 0:
            bounds.append([5., None])
        else:
            bounds.append([None, None])
for _ in range(len(crmsd_init)):
    bounds.append([None, None])
bounds.append([None, None])

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

stride = 3
log_q_md = log_q_md[::stride]
log_q_noise = log_q_noise[::stride]
    
N_md = float(len(log_q_md))
N_noise = float(len(log_q_noise))
    
angle_md_basis = angle_md_basis[::stride].to(device)
angle_noise_basis = angle_noise_basis[::stride].to(device)

dihedral_md_basis = dihedral_md_basis[::stride].to(device)
dihedral_noise_basis = dihedral_noise_basis[::stride].to(device)

LJ_md_basis = LJ_md_basis[::stride].to(device)
LJ_noise_basis = LJ_noise_basis[::stride].to(device)

rmsd_md_basis = rmsd_md_basis[::stride].to(device)
rmsd_noise_basis = rmsd_noise_basis[::stride].to(device)

bond_energy_md = bond_energy_md[::stride].to(device)
bond_energy_noise = bond_energy_noise[::stride].to(device)

log_q_md = torch.from_numpy(log_q_md).to(device)
log_q_noise = torch.from_numpy(log_q_noise).to(device)

LJ_basis_info = {}
for pair in pair_indices:
    i, j = pair
    with open(f"./output/{name}/LJ/LJ_basis/basis_info_{i}-{j}.pkl", 'rb') as file_handle:
        LJ_basis_info[(i,j)] = pickle.load(file_handle)

omega = np.array([LJ_basis_info[pair]['omega'] for pair in pair_indices])
omega = torch.from_numpy(omega).to(device)


def compute_loss_and_grad(x, weight_decay):
    x = torch.tensor(x, requires_grad = True, device = device)
    ca, cd, clj, crmsd, F = unflatten_parameters(x)

    energy_angle_md = torch.sum(ca*angle_md_basis, dim = (1,2))
    energy_dihedral_md = torch.sum(cd*dihedral_md_basis, dim = (1,2))
    energy_LJ_md = torch.sum(clj*LJ_md_basis, dim = (1,2))
    energy_rmsd_md = torch.sum(crmsd*rmsd_md_basis, dim = 1)
    
    energy_md = bond_energy_md + energy_angle_md + energy_dihedral_md + energy_LJ_md + energy_rmsd_md
    log_p_md = F - energy_md
    log_p_md = log_p_md + torch.log(torch.tensor([N_md/N_noise], device = device))

    energy_angle_noise = torch.sum(ca*angle_noise_basis, dim = (1,2))
    energy_dihedral_noise = torch.sum(cd*dihedral_noise_basis, dim = (1,2))
    energy_LJ_noise = torch.sum(clj*LJ_noise_basis, dim = (1,2))
    energy_rmsd_noise = torch.sum(crmsd*rmsd_noise_basis, dim = 1)
    
    energy_noise = bond_energy_noise + energy_angle_noise + energy_dihedral_noise + energy_LJ_noise + energy_rmsd_noise
    log_p_noise = F - energy_noise
    log_p_noise = log_p_noise + torch.log(torch.tensor([N_md/N_noise], device = device))

    logit = torch.stack(
        [ torch.cat([log_q_noise, log_q_md]),
          torch.cat([log_p_noise, log_p_md])]
        ).t()
    
    target = torch.cat([torch.zeros_like(log_q_noise), torch.ones_like(log_q_md)]).long()
    # class_weight = logit.new_tensor([N_md/(N_md + N_noise), N_noise/(N_md + N_noise)])
    # loss = torch.nn.functional.cross_entropy(logit, target, weight = class_weight, reduction = 'mean')
    loss = torch.nn.functional.cross_entropy(logit, target, reduction = 'mean')
    
    roughness = 0.5*torch.sum(clj[:,:,None]*omega*clj[:,None,:])
    loss = loss + weight_decay*roughness
    
    loss.backward()
    grad = x.grad.cpu().numpy()

    return loss.item(), grad

loss, grad = compute_loss_and_grad(c_init.numpy(), weight_decay = weight_decay)
x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 c_init.numpy(),
                                 args = [weight_decay],
                                 iprint = 1,
                                 factr = 100,
                                 pgtol = 1e-6,
                                 bounds = bounds)

ca, cd, clj, crmsd, F = unflatten_parameters(x)

bonded_parameters['reference_particle_3_angle']['alphas'] = ca[0,:]
bonded_parameters['reference_particle_3_angle']['a_over_the_range'] = angle_over_range
bonded_parameters['reference_particle_3_angle']['U'] = np.sum(angle_over_range_basis * ca[0,:], -1)

ca = ca[1:,:]
for i in range(ca.shape[0]):
    bonded_parameters['angle'][i]['alphas'] = ca[i,:]
    bonded_parameters['angle'][i]['a_over_the_range'] = angle_over_range    
    bonded_parameters['angle'][i]['U'] = np.sum(angle_over_range_basis * ca[i,:], -1)

for i in range(cd.shape[0]):
    bonded_parameters['dihedral'][i]['alphas'] = cd[i,:]
    bonded_parameters['dihedral'][i]['d_over_the_range'] = dihedral_over_range
    bonded_parameters['dihedral'][i]['U'] = np.sum(dihedral_over_range_basis * cd[i,:], -1)

LJ_parameters = {}
for i in range(len(pair_indices)):
    key = pair_indices[i]
    ulj = np.matmul(LJ_over_range_basis[key], clj[i])
    LJ_parameters[key] = {'U': ulj, 'r_over_range': r_over_range[key], 'r_min': r_over_range[key].min(), 'r_max': r_over_range[key].max()}

    
rmsd_parameters = {
    'rmsd_over_range': rmsd_over_range,
    'U': np.sum(rmsd_over_range_basis*crmsd, -1)
}

os.makedirs(f"./output/{name}/LJ/FF", exist_ok = True)    
with open(f"./output/{name}/LJ/FF/weight_decay_{weight_decay:.3E}.pkl", 'wb') as file_handle:
    pickle.dump({'bonded_parameters': bonded_parameters,
                 'LJ_parameters': LJ_parameters,
                 'rmsd_parameters': rmsd_parameters},
                file_handle)
    
    
