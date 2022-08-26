#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=learn_mmflow
#SBATCH --time=10:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/learn_mmflow_%A.txt

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import math
from sys import exit
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import transform, MMFlow, utils
sys.path.append("./script/flow/")
from functions import *
import pickle
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import simtk.openmm as omm
import simtk.openmm.app as app
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
parser.add_argument('--num_transforms', type = int, default = 16)
parser.add_argument('--hidden_size', type = int, default = 24)
args = parser.parse_args()

name = args.name
num_transforms = args.num_transforms
hidden_size = args.hidden_size
print(f"name: {name}, num_transforms: {num_transforms}, hidden_size: {hidden_size}")

data = torch.load(f"./output/{name}/md/ic.pt")
ic = data['ic']

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

helix_particle_index = set(range(2,14))
helix_angle_index = []
helix_angle_particle_index = []
helix_dihedral_index = []
helix_dihedral_particle_index = []
for i in range(len(coor_transformer.particle_visited_in_order)):
    p = coor_transformer.particle_visited_in_order[i]
    p1, p2, p3 = coor_transformer.angle_particle_idx[p]
    
    if p1 in helix_particle_index and \
       p2 in helix_particle_index and \
       p3 in helix_particle_index:
        helix_angle_index.append(i)
        helix_angle_particle_index.append([p1, p2, p3])

    p1, p2, p3, p4 = coor_transformer.dihedral_particle_idx[p]    
    if p1 in helix_particle_index and \
       p2 in helix_particle_index and \
       p3 in helix_particle_index and \
       p4 in helix_particle_index:
        helix_dihedral_index.append(i)
        helix_dihedral_particle_index.append([p1, p2, p3, p4])
        
helix_angle = ic.angle[:, helix_angle_index]
helix_dihedral = ic.dihedral[:, helix_dihedral_index]

helix_reference_particle_3_angle_flag = False

if coor_transformer.ref_particle_1 in helix_particle_index and \
   coor_transformer.ref_particle_2 in helix_particle_index and \
   coor_transformer.ref_particle_3 in helix_particle_index:
    helix_angle = torch.cat(
        [ic.reference_particle_3_angle[:, None],
         helix_angle], dim = -1)
    helix_reference_particle_3_angle_flag = True

with open(f"./output/{name}/flow/helix_index.pkl", 'wb') as file_handle:
    pickle.dump({'helix_particle_index': helix_particle_index,
                 'helix_angle_index': helix_angle_index,
                 'helix_angle_particle_index': helix_angle_particle_index,
                 'helix_dihedral_index': helix_dihedral_index,
                 'helix_dihedral_particle_index': helix_dihedral_particle_index,
                 'helix_reference_particle_3_angle_flag': helix_reference_particle_3_angle_flag},
                file_handle)

helix_angle = helix_angle.double()
helix_angle = helix_angle / math.pi
helix_angle = torch.clamp(helix_angle, min = 0.0, max = 1.0)
helix_dihedral = helix_dihedral.double()
helix_dihedral = torch.clamp(helix_dihedral, min = -math.pi, max = math.pi)

feature = torch.cat([helix_angle, helix_dihedral], dim = -1)
circular_feature_flag = torch.tensor(
    [False for _ in range(helix_angle.shape[-1])] +
    [True for _ in range(helix_dihedral.shape[-1])])

feature_size = feature.shape[-1]
context_size = None

transform_feature_flag = torch.zeros(num_transforms, feature_size)
for j in range(feature_size):
    tmp = torch.as_tensor([i % 2 for i in range(num_transforms)])
    tmp = tmp[torch.randperm(num_transforms)]    
    transform_feature_flag[:, j] = tmp
    
num_blocks = 2
conditioner_net_create_fn = lambda feature_size, context_size, output_size: \
    transform.ResidualNet(feature_size,
                          context_size,
                          output_size,
                          hidden_size = hidden_size,
                          num_blocks = num_blocks)

num_bins_regular = 4
num_bins_circular = 4
mmflow = MMFlow(feature_size, context_size,
                circular_feature_flag,
                transform_feature_flag,
                conditioner_net_create_fn,
                num_bins_circular = num_bins_circular,
                num_bins_regular = num_bins_regular)

## split data into training and validation
num_samples = feature.shape[0]
num_train_samples = int(num_samples * 0.9)

train_sample_indices = set(np.random.choice(num_samples,
                                            size = num_train_samples,
                                            replace = False))
train_sample_flag = np.array([ i in train_sample_indices for i in range(num_samples)])

context = None
feature_train = feature[train_sample_flag]
feature_validation = feature[~train_sample_flag]

batch_size = 1024*8

dataset = torch.utils.data.TensorDataset(feature_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')    
mmflow = mmflow.to(device)

optimizer = optim.Adam(mmflow.parameters(), lr=1e-3)
num_epochs = 100

loss_train_record = []
loss_validation_record = []

start_time = time.time()
for idx_epoch in range(num_epochs):
    for idx_step, data in enumerate(dataloader, 0):
        mmflow.train()
        lr = optimizer.param_groups[0]['lr']
        batch_feature = data[0]
        batch_feature = batch_feature.to(device)
        
        log_density = mmflow.compute_log_prob(batch_feature, None)
        loss = -torch.mean(log_density)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_train_record.append(loss.item())

        if (idx_step + 1) % 10 == 0:
            print(f"step: {idx_step:>5}, lr: {lr:.3E}, loss: {loss.item():.3f}", flush = True)
        # if (idx_step + 1) % 100 == 0:
        #     print("time used for 100 steps: {:.3f}".format(time.time() - start_time), flush = True)
        #     start_time = time.time()

    mmflow.eval()
    log_density_list = []
    with torch.no_grad():
        num_batches_validation = feature_validation.shape[0]//batch_size + 1
        for idx_batch in range(num_batches_validation):
            batch_feature = feature_validation[idx_batch*batch_size: (idx_batch+1)*batch_size].to(device)
            log_density = mmflow.compute_log_prob(batch_feature, None)
            log_density_list.append(log_density)
        log_density = torch.cat(log_density_list)
        loss = -torch.mean(log_density.cpu())
        loss_validation_record.append(loss.item())
        
    print(f"idx_epoch: {idx_epoch:>5}, validation_loss: {loss.item():.3f}", flush = True)

    os.makedirs(f"./output/{name}/flow/model/hidden_size_{hidden_size}_num_transforms_{num_transforms}", exist_ok = True)
    torch.save({'feature_size': feature_size,
                'context_size': context_size,
                'circular_feature_flag': circular_feature_flag,
                'transform_feature_flag': transform_feature_flag,
                'train_sample_flag': train_sample_flag,
                'hidden_size': hidden_size,
                'num_blocks': num_blocks,
                'num_transforms': num_transforms,
                'num_bins_circular': num_bins_circular,
                'num_bins_regular': num_bins_regular,
                'loss_train_record': loss_train_record,
                'loss_validation_record': loss_validation_record,
                'state_dict': mmflow.state_dict()},
               f"./output/{name}/flow/model/hidden_size_{hidden_size}_num_transforms_{num_transforms}/epoch_{idx_epoch}.pt")

    print(f"time used for one epoch of traing: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
exit()
