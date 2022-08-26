#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=make_system
#SBATCH --time=1:00:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-11
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/make_system_%a.txt

import numpy as np
import openmm as omm
import openmm.unit as unit
import openmm.app as app
import math
import mdtraj
import pickle
from sys import exit
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG/")
from CLCG.utils.make_system import *
import argparse
import pandas as pd
import os
import torch
torch.set_default_dtype(torch.float64)
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import transform, MMFlow, utils
import torch.distributions as distributions
from openmmtorch import TorchForce
import time
from copy import copy

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
parser.add_argument('--num_transforms', type = int, default = 16)
parser.add_argument('--hidden_size', type = int, default = 24)
args = parser.parse_args()

name = args.name
num_transforms = args.num_transforms
hidden_size = args.hidden_size

protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
# job_idx = 1
# name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
AA_info = pd.read_csv("./script/md/amino_acids_with_learned_sigmas.csv", index_col = 'name')

masses = [AA_info.loc[r.name, 'mass'] for r in psf.residues]

with open(f"./output/{name}/im/FF/bonded_parameters.pkl", 'rb') as file_handle:
    bonded_parameters = pickle.load(file_handle)

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

protein_info = pd.read_csv('./script/md/protein_temperature_and_ionic_strength.txt', comment = "#", index_col = 0)
helix_particle_index = set(range(2,14))
system = make_system(masses, coor_transformer, bonded_parameters, protein_info.loc[name, 'temperature'],
                     exclude_angle_particle_index = helix_particle_index,
                     exclude_dihedral_particle_index = helix_particle_index)

class FlowModel(torch.nn.Module):
    def __init__(self, T, angle_index, dihedral_index, mmflow):
        super(FlowModel, self).__init__()
        self.T = torch.nn.Parameter(T, requires_grad = False)
        
        self.angle_index = torch.nn.Parameter(
            angle_index,
            requires_grad = False)
        
        self.dihedral_index = torch.nn.Parameter(
            dihedral_index,
            requires_grad = False)

        self.mmflow = mmflow

        self.Kb = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self.Kb = self.Kb.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)

        
    def forward(self, positions):
        xyz = torch.unsqueeze(positions, 0)
        
        ## compute angles
        xyz_i = torch.index_select(xyz, 1, self.angle_index[:, 0])
        xyz_j = torch.index_select(xyz, 1, self.angle_index[:, 1])
        xyz_k = torch.index_select(xyz, 1, self.angle_index[:, 2])    

        v = xyz_i - xyz_j
        w = xyz_k - xyz_j

        v = v / torch.sqrt(torch.sum(v**2, -1, keepdim = True))
        w = w / torch.sqrt(torch.sum(w**2, -1, keepdim = True))

        inner_product = torch.sum(v*w, -1)
        angles = torch.acos(inner_product)
        
        xyz_i = torch.index_select(xyz, 1, self.dihedral_index[:, 0])
        xyz_j = torch.index_select(xyz, 1, self.dihedral_index[:, 1])
        xyz_k = torch.index_select(xyz, 1, self.dihedral_index[:, 2])
        xyz_l = torch.index_select(xyz, 1, self.dihedral_index[:, 3])

        b1 = xyz_i - xyz_j
        b2 = xyz_j - xyz_k
        b3 = xyz_l - xyz_k

        b1_cross_b2 = torch.cross(b1, b2)
        b3_cross_b2 = torch.cross(b3, b2)

        cos_d = torch.norm(b2, dim = -1)*torch.sum(b1_cross_b2*b3_cross_b2, -1)
        sin_d = torch.sum(b2*torch.cross(b3_cross_b2, b1_cross_b2), -1)

        dihedrals = torch.atan2(sin_d, cos_d)

        feature = torch.cat([angles/math.pi, dihedrals], dim = -1)

        z, logabsdet = self.mmflow(feature, None)
        base_dist = distributions.Uniform(low = self.mmflow.base_dist_low,
                                          high = self.mmflow.base_dist_high)
        base_dist = distributions.Independent(base_dist, 1)
        log_prob = base_dist.log_prob(z) + logabsdet + torch.log(angles.new_tensor(1./math.pi))*angles.shape[-1]
        
        U = -self.Kb*self.T*log_prob
        
        return U
    
helix_angle_particle_index = []
helix_dihedral_particle_index = []
for i in range(len(coor_transformer.particle_visited_in_order)):
    p = coor_transformer.particle_visited_in_order[i]
    p1, p2, p3 = coor_transformer.angle_particle_idx[p]
    if p1 in helix_particle_index and \
       p2 in helix_particle_index and \
       p3 in helix_particle_index:
        helix_angle_particle_index.append([p1, p2, p3])

    p1, p2, p3, p4 = coor_transformer.dihedral_particle_idx[p]    
    if p1 in helix_particle_index and \
       p2 in helix_particle_index and \
       p3 in helix_particle_index and \
       p4 in helix_particle_index:
        helix_dihedral_particle_index.append([p1, p2, p3, p4])
        
if coor_transformer.ref_particle_1 in helix_particle_index and \
   coor_transformer.ref_particle_2 in helix_particle_index and \
   coor_transformer.ref_particle_3 in helix_particle_index:
    helix_angle_particle_index = [[coor_transformer.ref_particle_2,
                                   coor_transformer.ref_particle_1,
                                   coor_transformer.ref_particle_3]] + helix_angle_particle_index

## load mmflow model
max_epoch = 99
data = torch.load(
    f"./output/{name}/flow/model/hidden_size_{hidden_size}_num_transforms_{num_transforms}/epoch_{max_epoch}.pt",
    map_location = torch.device('cpu')
)
idx_epoch = np.argmin(data['loss_validation_record'])
data = torch.load(
    f"./output/{name}/flow/model/hidden_size_{hidden_size}_num_transforms_{num_transforms}/epoch_{idx_epoch}.pt",
    map_location = torch.device('cpu')    
)

conditioner_net_create_fn = lambda feature_size, context_size, output_size: \
    transform.ResidualNet(feature_size,
                          context_size,
                          output_size,
                          hidden_size = data['hidden_size'],
                          num_blocks = data['num_blocks'])
mmflow = MMFlow(data['feature_size'],
                data['context_size'],
                data['circular_feature_flag'],
                data['transform_feature_flag'],
                conditioner_net_create_fn,
                num_bins_circular = data['num_bins_circular'],
                num_bins_regular = data['num_bins_regular'])

mmflow.load_state_dict(data['state_dict'])

angle_index = torch.LongTensor(helix_angle_particle_index)
dihedral_index = torch.LongTensor(helix_dihedral_particle_index)

T = protein_info.loc[name, 'temperature']
flow_model = FlowModel(torch.tensor(T), angle_index, dihedral_index, mmflow)
traj = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf, stride = 1)
xyz = traj.xyz
U = flow_model(torch.from_numpy(xyz[0]).double())

traced_flow_model = torch.jit.trace(flow_model, torch.from_numpy(xyz[0]).double())
os.makedirs(f"./output/{name}/im_flow", exist_ok = True)
traced_flow_model.save(f"./output/{name}/im_flow/traced_flow_model.pt")
torch_force = TorchForce(f"./output/{name}/im_flow/traced_flow_model.pt")
system.addForce(torch_force)

xml = omm.XmlSerializer.serialize(system)
f = open(f"./output/{name}/im_flow/system.xml", 'w')
f.write(xml)
f.close()

exit()

fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)
xyz_init = xyz[100]
context.setPositions(xyz_init)

os.makedirs(f"./output/{name}/im_flow/CG_simulations/", exist_ok = True)
file_handle = open(f"./output/{name}/im_flow/CG_simulations/traj.dcd", 'wb')
dcd_file = app.DCDFile(file_handle, psf.to_openmm(), dt = 200*unit.femtoseconds)

start_time = time.time()
num_frames = len(traj)*100

num_frames = 30
for i in range(num_frames):
    integrator.step(100)
    state = context.getState(getPositions = True)
    pos = state.getPositions()
    dcd_file.writeModel(pos)

    if (i + 1) % 1 == 0:
        print(i, flush = True)

print("time used: {:.2f}".format(time.time() - start_time))
file_handle.close()

exit()
