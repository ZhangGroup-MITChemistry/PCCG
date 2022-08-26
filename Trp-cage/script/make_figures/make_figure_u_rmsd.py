import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
import numpy as np
import torch
import pyemma
import mdtraj
import os
from scipy.stats import gaussian_kde
from sys import exit
from matplotlib import cm

name = "2JOF"
include_rmsd = True
weight_decay = 2e-7

data = torch.load(f"./output/{name}/FF/FF_rmsd_{include_rmsd}_weight_decay_{weight_decay:.3E}.pt")
FF = data['FF']
bonded_ff, lj_ff, rmsd_ff = FF['bonded'], FF['lj'], FF['rmsd']

fig = plt.figure()
fig.clf()
rmsd = np.linspace(rmsd_ff['min'].item(), rmsd_ff['max'].item(), rmsd_ff['U'].shape[1])
plt.plot(rmsd, rmsd_ff['U'][0,:].numpy(), linewidth = 2, color = 'black')
plt.savefig(f"./output/plots/{name}_u_rmsd.eps")

