#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=CL
#SBATCH --time=10:00:00
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --array=0-3
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/cl_%a.txt

import argparse
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import pickle
from scipy import optimize
from sys import exit
import mdtraj
import pandas as pd
import openmm.unit as unit
from FastMBAR import *
import sys
import MMFlow
from MMFlow import utils
import pyarrow.feather as feather
import os
import math
import torch.nn as nn
from functools import reduce
from scipy.sparse import coo_matrix, csr_matrix
import itertools
import argparse
import ray
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--weight_decay", type = float)
parser.add_argument("--elec_type", type = str)
parser.add_argument("--ss_type", type = str)
#parser.add_argument("--bounds_flag", type = bool)
args = parser.parse_args()

weight_decay = args.weight_decay
ss_type = args.ss_type
elec_type = args.elec_type
#bounds_flag = args.bounds_flag
print(f"weight_decay: {weight_decay:.3E}")
print(f"elec_type: {elec_type}")
print(f"ss_type: {ss_type}")
#print(f"bounds_flag: {bounds_flag}")

protein_names = pd.read_csv("./info/protein_names.txt",
                            comment = "#",
                            header = None)
protein_names = protein_names.iloc[:, 0].tolist()

@ray.remote
class CL_Actor(torch.nn.Module):
    def __init__(self, name):
        #super(CL_Actor, self).__init__()
        torch.set_default_dtype(torch.float64)        
        super().__init__()        
        self.name = name
        
        protein_info = pd.read_csv(
            "./info/protein_temperature_and_ionic_strength.txt",
            index_col = 'name',
            comment = '#')
        self.T = protein_info.loc[self.name, 'temperature']
        
        ic_imus = torch.load(f"./output/{name}/ic_imus.pt")['ic']
        ic_md = torch.load(f"./output/{name}/ic_md.pt")['ic']

        self.N_imus = len(ic_imus)
        self.N_md = len(ic_md)
        self.N = self.N_imus + self.N_md
        
        self.target = torch.cat([torch.zeros(self.N_imus), torch.ones(self.N_md)])
        
        bond_imus = torch.cat(
            (ic_imus.reference_particle_2_bond[:, None],
             ic_imus.reference_particle_3_bond[:, None],
             ic_imus.bond), dim = -1)
        bond_md = torch.cat(
            (ic_md.reference_particle_2_bond[:, None],
             ic_md.reference_particle_3_bond[:, None],
             ic_md.bond), dim = -1)

        bond = torch.cat([bond_imus, bond_md])
        del ic_imus, ic_md, bond_imus, bond_md

        data = torch.load(f'./output/{name}/basis_angle_and_dihedrals.pt')
        self.rp3angle = torch.cat(
            [data['imus']['reference_particle_3_angle'],
             data['md']['reference_particle_3_angle']])
        self.angle = torch.cat(
            [data['imus']['angle'],
             data['md']['angle']])
        self.dihedral = torch.cat(
            [data['imus']['dihedral'],
             data['md']['dihedral']])
        
        self.angle_grid = data['grid']['angle_basis']
        self.angle_min = data['grid']['angle_min']        
        self.angle_max = data['grid']['angle_max']
        
        self.dihedral_grid = data['grid']['dihedral_basis']
        self.dihedral_min = data['grid']['dihedral_min']        
        self.dihedral_max = data['grid']['dihedral_max']
        del data

        with open(f"./output/{name}/bonded_parameters_im.pkl", 'rb') as file_handle:
            bonded_parameters = pickle.load(file_handle)
            self.bonded_parameters = bonded_parameters
            
        b0 = torch.cat([torch.tensor([bonded_parameters['reference_particle_2_bond']['b0']]),
                        torch.tensor([bonded_parameters['reference_particle_3_bond']['b0']]),
                        bonded_parameters['bond']['b0']])
        kb = torch.cat([torch.tensor([bonded_parameters['reference_particle_2_bond']['kb']]),
                        torch.tensor([bonded_parameters['reference_particle_3_bond']['kb']]),
                        bonded_parameters['bond']['kb']])
        bond_energy = torch.sum(0.5*kb*(bond - b0)**2, -1)
        self.register_buffer('bond_energy', bond_energy, persistent = False)
        
        self.c3a = nn.Parameter(bonded_parameters['reference_particle_3_angle']['alphas'])
        self.ca = nn.Parameter(torch.cat([p['alphas'] for p in bonded_parameters['angle']]))
        self.cd = nn.Parameter(torch.cat([p['alphas'] for p in bonded_parameters['dihedral']]))

        data = torch.load(f"./output/{name}/basis_lj.pt")
        self.lj = torch.cat([data['imus'], data['md']])
                
        data = torch.load(f"./output/common/basis_lj_grid.pt")
        self.lj_grid = data['basis']
        self.lj_min = data['min']
        self.lj_max = data['max']

        lj_omega = data['omega']
        self.register_buffer('lj_omega', lj_omega, persistent = False)
        self.n_lj, self.nb_lj = lj_omega.shape[0], lj_omega.shape[1]
        clj = torch.zeros((self.n_lj, self.nb_lj))
        clj[:, 0] = 5.0
        clj = clj.view(-1)
        self.clj = nn.Parameter(clj)        
        #self.clj = nn.Parameter(torch.zeros(self.lj.shape[-1]))
        del data

        if os.path.exists(f"./output/{name}/basis_rmsd_{ss_type}.pt"):
            data = torch.load(f"./output/{name}/basis_rmsd_{ss_type}.pt")
            rmsd_imus = torch.cat([d['imus'] for d in data], dim = -1)
            rmsd_md = torch.cat([d['md'] for d in data], dim = -1)
            self.rmsd = torch.cat([rmsd_imus, rmsd_md])
            self.rmsd_grid = torch.stack([d['grid']['basis'] for d in data])
            self.rmsd_max = torch.tensor([d['grid']['max'] for d in data])            
            self.rmsd_min = torch.tensor([d['grid']['min'] for d in data])

            self.n_rmsd = len(data)
            self.nb_rmsd = self.rmsd_grid.shape[-1]
            del data, rmsd_imus, rmsd_md
            self.crmsd = nn.Parameter(torch.zeros(self.rmsd.shape[-1]))
            
        else:
            self.rmsd = None
            self.rmsd_min = None
            self.rmsd_max = None            
            self.n_rmsd = 0
            self.nb_rmsd = 0
            self.crmsd = nn.Parameter(torch.tensor([]))

        self.rmsd = None
        self.rmsd_min = None
        self.rmsd_max = None            
        self.n_rmsd = 0
        self.nb_rmsd = 0
        self.crmsd = nn.Parameter(torch.tensor([]))

            
        data = torch.load(f"./output/{name}/elec_energy_{elec_type}.pt")
        elec_energy = torch.cat([data['imus'], data['md']])
        self.register_buffer('elec_energy', elec_energy, persistent = False)
        del data

        data = torch.load(f"./output/{name}/log_q_imus.pt")
        self.log_q = torch.cat([data['imus'], data['md']])

        self._numel_cache = None

        ## initialize self.F
        with torch.no_grad():
            u_p = self.bond_energy + self.elec_energy + \
                  self.rp3angle.mv(self.c3a) + self.angle.mv(self.ca) + \
                  self.dihedral.mv(self.cd)
        u_p = u_p.numpy()
        u_q = -self.log_q.numpy()

        energy_matrix = np.vstack([u_q, u_p])
        num_conf = np.array([self.N_imus, self.N_md], dtype = np.float64)
        fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True)
        
        self.F = nn.Parameter(torch.tensor([fastmbar.F[-1]]))
        
        # ## Because pytorch 1.10 has limited support on sparse csr matrix,
        # ## we use scipy.sparse.csr_matrix for now
        # self.rp3angle = self._torch_coo_to_scipy_csr(self.rp3angle)        
        # self.angle = self._torch_coo_to_scipy_csr(self.angle)
        # self.dihedral = self._torch_coo_to_scipy_csr(self.dihedral)
        # self.lj = self._torch_coo_to_scipy_csr(self.lj)                

        print(f"finish initializing {name}", flush = True)
        
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self.parameters(), 0)
        return self._numel_cache

    def get_named_parameter_numel(self):
        res = OrderedDict()
        for name, p in self.named_parameters():
            res[name] = p.numel()
        return res
    
    def _gather_flat_grad(self):
        views = []
        for p in self.parameters():
            if p.grad is None:
                view =  p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_param(self):
        views = []
        for p in self.parameters():
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _set_param(self, update):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(update[offset:offset + numel].view_as(p))
            offset += numel
        assert offset == self._numel()                

    def _torch_coo_to_scipy_csr(self, s):
        s = s.coalesce()
        row, col = s.indices().tolist()
        data = s.values().numpy()
        shape = list(s.shape)
        return coo_matrix((data, (row, col)), shape = shape).tocsr()

    def _scipy_csr_to_torch_coo(self, s):
        s = s.tocoo()
        s = torch.sparse_coo_tensor(np.array([s.row, s.col]), s.data, size = s.shape)
        return s

    def make_parameter_bounds(self):
        bounds = []
        for name, param in self.named_parameters():
            if name == 'clj':
                for i,j in itertools.product(range(self.n_lj), range(self.nb_lj)):
                    if j == 0:
                        bounds.append([0, None])
                    else:
                        bounds.append([None, None])
            else:
                bounds += [[None, None] for _ in range(param.numel())]
        assert(len(bounds) == self._numel())
        return bounds

    def compute_loss_and_grad(self, x, weight_decay):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
            
        device = self.ca.device
        x = x.to(device)
        
        self._set_param(x)
        self.zero_grad()

        self.rp3angle = self.rp3angle.to(device)
        self.angle = self.angle.to(device)
        self.dihedral = self.dihedral.to(device)
        self.lj = self.lj.to(device)        
        self.log_q = self.log_q.to(device)
        self.target = self.target.to(device)

        kbT = unit.BOLTZMANN_CONSTANT_kB * self.T * unit.kelvin * unit.AVOGADRO_CONSTANT_NA
        kbT_in_kJ_per_mole = kbT.value_in_unit(unit.kilojoule_per_mole)

        ## note all energy terms except LJ are in reduced units
        ## at the corresponding temperature.
        ## LJ energy term has the unit of kcal/mol
        u = self.bond_energy + self.elec_energy + \
            self.rp3angle.mv(self.c3a) + self.angle.mv(self.ca) + \
            self.dihedral.mv(self.cd) + self.lj.mv(self.clj)/kbT_in_kJ_per_mole
        
        if self.rmsd is not None:
            self.rmsd = self.rmsd.to(device)                    
            u = u + self.rmsd.mv(self.crmsd)
                
        log_p = - (u - self.F) + u.new_tensor([self.N_md/self.N_imus])
        logit = log_p - self.log_q
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, self.target, reduction = 'mean')
        loss.backward()
        loss = loss.cpu().item()
        
        clj = self.clj.view(self.n_lj, self.nb_lj)
        loss_wc = 0.5*weight_decay*torch.sum(clj[:,None,:].matmul(self.lj_omega).matmul(clj[:,:,None]))
        loss_wc.backward()
        
        loss += loss_wc.cpu().item()
        grad = self._gather_flat_grad()

        return loss, grad.cpu().numpy()
    
    def compute_loss_and_grad_in_batch(self, x, weight_decay, batch_size):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
            
        device = self.ca.device
        x = x.to(device)
        
        self._set_param(x)

        if batch_size == -1:
            batch_size = self.N
            
        n_batch = self.N // batch_size
        if self.N % batch_size != 0: n_batch += 1

        tot_loss = 0
        self.zero_grad()        
        for k in range(n_batch):
            u_bond = self.bond_energy[k*batch_size:(k+1)*batch_size]
            u_elec = self.elec_energy[k*batch_size:(k+1)*batch_size]

            rp3angle = self._scipy_csr_to_torch_coo(self.rp3angle[k*batch_size:(k+1)*batch_size]).to(device)        
            angle = self._scipy_csr_to_torch_coo(self.angle[k*batch_size:(k+1)*batch_size]).to(device)
            dihedral = self._scipy_csr_to_torch_coo(self.dihedral[k*batch_size:(k+1)*batch_size]).to(device)
            lj = self._scipy_csr_to_torch_coo(self.lj[k*batch_size:(k+1)*batch_size]).to(device)

            u = u_bond + u_elec + rp3angle.mv(self.c3a) + angle.mv(self.ca) + \
                dihedral.mv(self.cd) + lj.mv(self.clj)

            if self.rmsd is not None:
                rmsd = self.rmsd[k*batch_size:(k+1)*batch_size].to(device)
                u = u + rmsd.mv(self.crmsd)
                
            log_p = - (u - self.F) + u.new_tensor([self.N_md/self.N_imus])
            log_q = self.log_q[k*batch_size:(k+1)*batch_size].to(device)
            target = self.target[k*batch_size:(k+1)*batch_size].to(device)

            logit = log_p - log_q
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, target, reduction = 'sum')        
            loss.backward()
            tot_loss += loss.cpu().item()
            
        loss = tot_loss / self.N
        for p in self.parameters():
            if p.grad is not None:
                p.grad = p.grad / self.N

        clj = self.clj.view(self.n_lj, self.nb_lj)
        loss_wc = 0.5*weight_decay*torch.sum(clj[:,None,:].matmul(self.lj_omega).matmul(clj[:,:,None]))
        loss_wc.backward()
        
        loss += loss_wc.cpu().item()
        grad = self._gather_flat_grad()

        return loss, grad.cpu().numpy()

    def get_FF(self):
        self.bonded_parameters['reference_particle_3_angle']['U'] = torch.matmul(self.angle_grid,
                                                                                 self.c3a.cpu().data)
        nb_angle = self.angle_grid.shape[-1]                
        for i in range(len(self.bonded_parameters['angle'])):
            ca = self.ca.cpu().data[i*nb_angle:(i+1)*nb_angle]
            self.bonded_parameters['angle'][i]['U'] = torch.matmul(self.angle_grid, ca)
            
        nb_dihedral = self.dihedral_grid.shape[-1]        
        for i in range(len(self.bonded_parameters['dihedral'])):
            cd = self.cd.cpu().data[i*nb_dihedral:(i+1)*nb_dihedral]
            self.bonded_parameters['dihedral'][i]['U'] = torch.matmul(self.dihedral_grid, cd)

        clj = self.clj.cpu().data.view(self.n_lj, 1, self.nb_lj)
        ulj = torch.sum(self.lj_grid*clj, dim = -1)

        urmsd = None

        if self.rmsd is not None:
            crmsd = self.crmsd.cpu().data.view(self.n_rmsd, 1, self.nb_rmsd)
            urmsd = torch.sum(self.rmsd_grid*crmsd, dim = -1)
            
        FF = {'bonded': self.bonded_parameters,
              'lj': {'U': ulj, 'min': self.lj_min, 'max': self.lj_max},
              'rmsd': {'U': urmsd, 'min': self.rmsd_min, 'max': self.rmsd_max}}
        
        return FF

    def get_extra_state(self):
        return None
    
def gather_param(param_list, param_numel, clj_sum = False):
    assert(len(param_list) == len(param_numel))
    res = []
    clj = []
    for i in range(len(param_numel)):
        p = param_list[i]
        p_numel = param_numel[i]

        offset = 0
        for k, v in p_numel.items():
            if k != 'clj':
                res.append(p[offset:offset+v])
            else:
                clj.append(p[offset:offset+v])
            offset += v

    if clj_sum is False:
        res.append(clj[0])
    else:
        res.append(torch.sum(torch.stack(clj), dim = 0))
    return torch.cat(res, 0)

def split_param(x, param_numel):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    
    p_list = []
    offset = 0
    for i in range(len(param_numel)):
        p_numel = param_numel[i]
        p = []

        for k, v in p_numel.items():
            if k != 'clj':
                p.append(x[offset:offset+v])
                offset += v
            else:
                p.append(x[-v:])
        p = torch.cat(p, 0)
        p_list.append(p)
        
    return p_list

#ray.shutdown()
ray.init(ignore_reinit_error = True, _temp_dir = "/home/gridsan/dingxq/tmp/ray")    
actors = [CL_Actor.options(num_cpus = 20, num_gpus = 1).remote(name) for name in protein_names]
_ = [actor.cuda.remote() for actor in actors]
param_numel = ray.get([actor.get_named_parameter_numel.remote() for actor in actors])
param_list = ray.get([actor._gather_flat_param.remote() for actor in actors])

def compute_loss_and_grad(x, weight_decay):
    p_list = split_param(x, param_numel)    
    res = ray.get([actors[i].compute_loss_and_grad.remote(p_list[i], weight_decay/len(actors)) for i in range(len(actors))])
    loss = np.sum([l for l, g in res])
    grad_list = [torch.from_numpy(g.copy()) for l, g in res]
    grad = gather_param(grad_list, param_numel, True).numpy()    
    return loss, grad

x_init = gather_param(param_list, param_numel).cpu().numpy()
loss, grad = compute_loss_and_grad(x_init, weight_decay)
bounds = ray.get(actors[0].make_parameter_bounds.remote())
offset = 0
for k, v in param_numel[0].items():
    if k == 'clj':
        clj_bounds = bounds[offset:offset+v]
        break
    else:
        offset += v

bounds = [[None, None] for i in range(len(x_init))]
bounds[-len(clj_bounds):] = clj_bounds

x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 x_init,
                                 args = [weight_decay],
                                 bounds = bounds,
                                 iprint = 1,
                                 pgtol = 1e-5,
                                 factr = 100,
                                 maxfun = 50_000,
                                 maxiter = 50_000)

FFs = ray.get([actor.get_FF.remote() for actor in actors])
state_dicts = ray.get([actor.state_dict.remote() for actor in actors])
    
param_list = ray.get([actor._gather_flat_param.remote() for actor in actors])
res = ray.get([actors[i].compute_loss_and_grad.remote(param_list[i], 0.0) for i in range(len(actors))])
loss = [l for l, g in res]
loss_total = np.sum(loss)

print("loss for each proteins", loss)
print(f"total_loss without weight decay: {loss_total:.3f}")

for i in range(len(protein_names)):
    name = protein_names[i]
    FF = FFs[i]
    state_dict = state_dicts[i]
    os.makedirs(f"./output/{name}/FF_wo_rmsd_bound_0", exist_ok = True)
    torch.save({'FF': FF, 'state_dict': state_dict, 'loss': loss[i], 'loss_total': loss_total},
               f"./output/{name}/FF_wo_rmsd_bound_0/FF_elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.pt")
    print(name)


exit()
