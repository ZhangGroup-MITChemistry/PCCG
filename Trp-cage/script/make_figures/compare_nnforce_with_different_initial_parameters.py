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
import matplotlib.pyplot as plt

name = '2JOF'
full_include_rmsd = False
full_weight_decay = 4e-7

print(f"full rmsd: {full_include_rmsd}, weight_decay: {full_weight_decay:.3E}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
hidden_size = 32
n = psf.n_atoms
index = torch.LongTensor([[i,j] for i in range(psf.n_atoms) for j in range(i+4, psf.n_atoms)])
nnforce_net_1 = NNForceNet(n - 2,
                           n - 3,
                           len(index),
                           hidden_size)

nnforce_net_2 = NNForceNet(n - 2,
                           n - 3,
                           len(index),
                           hidden_size)

weight_decay = 1e-3
data = torch.load(f"./output/{name}/FF/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_NNForce_weight_decay_{weight_decay:.3E}_different_initial_parameters.pth")
state_dict = data['state_dict']
nnforce_net_1.load_state_dict(state_dict)

data = torch.load(f"./output/{name}/FF/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_NNForce_weight_decay_{weight_decay:.3E}.pth")
state_dict = data['state_dict']
nnforce_net_2.load_state_dict(state_dict)


p1 = torch.cat([p.data.flatten() for p in nnforce_net_1.parameters()])
p2 = torch.cat([p.data.flatten() for p in nnforce_net_2.parameters()])

fig = plt.figure()
fig.clf()
plt.plot(p1.numpy(), p2.numpy(), '.', alpha = 0.5)
plt.savefig('./output/plots/compare_nnforce_parameters_learned_with_different_initial_parameters.png')
