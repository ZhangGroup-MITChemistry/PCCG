#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2019/11/26 16:13:12

#SBATCH --job-name=sample
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --open-mode=truncate
#SBATCH --array=1
#SBATCH --mem=10G
#SBATCH --output=./slurm_output/sample_%a.txt

import pickle
import math
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch import distributions
import mdtraj
import simtk.openmm as omm
import simtk.openmm.app as app
import simtk.unit as unit
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import transform, MMFlow, utils
sys.path.append("./script/flow/")
from functions import *
from sys import exit
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
parser.add_argument('--num_transforms', type = int, default = 16)
parser.add_argument('--hidden_size', type = int, default = 24)
args = parser.parse_args()

name = args.name
num_transforms = args.num_transforms
hidden_size = args.hidden_size

print(f"name: {name}, num_transforms: {num_transforms}, hidden_size: {hidden_size}")

## load mmflow model
max_epoch = 99
data = torch.load(f"./output/{name}/flow/model/hidden_size_{hidden_size}_num_transforms_{num_transforms}/epoch_{max_epoch}.pt")
idx_epoch = np.argmin(data['loss_validation_record'])
data = torch.load(f"./output/{name}/flow/model/hidden_size_{hidden_size}_num_transforms_{num_transforms}/epoch_{idx_epoch}.pt")

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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
mmflow = mmflow.to(device)

torch.manual_seed(1000)
data = torch.load(f"./output/{name}/md/ic.pt")
ic = data['ic']
num_samples = len(ic)

batch_size = 1024*4

feature_flow_list = []
for idx_batch in range(num_samples//batch_size + 2):
    print(f"idx_batch: {idx_batch}")
    with torch.no_grad():
        feature_flow, _ = mmflow.sample_and_compute_log_prob(batch_size, None)
    feature_flow = torch.squeeze(feature_flow)
    feature_flow_list.append(feature_flow.cpu())
    
feature_flow = torch.cat(feature_flow_list, 0)
flag = torch.all(~torch.isnan(feature_flow), -1)
feature_flow = feature_flow[flag]
feature_flow = feature_flow[0:num_samples]

torch.save(feature_flow, f'./output/{name}/flow/feature_from_flow.pt')

exit()

ic_flow, _ = feature_and_context_to_ic(context_flow, feature_flow)

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

xyz_flow, logabsdet = coor_transformer.compute_xyz_from_internal_coordinate(
    ic_flow.reference_particle_1_xyz,
    ic_flow.reference_particle_2_bond,
    ic_flow.reference_particle_3_bond,
    ic_flow.reference_particle_3_angle,
    ic_flow.bond,
    ic_flow.angle,
    ic_flow.dihedral
)
ic_logabsdet = -logabsdet

torch.save({'ic':ic_flow, 'ic_logabsdet': ic_logabsdet}, f"./output/{name}/flow/ic.pt")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_flow = mdtraj.Trajectory(xyz_flow.numpy(), psf)
traj_flow.save_dcd(f"./output/{name}/flow/traj.dcd")

exit()
