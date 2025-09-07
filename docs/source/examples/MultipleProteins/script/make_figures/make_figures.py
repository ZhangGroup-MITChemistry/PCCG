import mdtraj
import argparse
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
from matplotlib.backends.backend_pdf import PdfPages
from sys import exit
from itertools import product
import pandas as pd
import ray
import os

parser = argparse.ArgumentParser()
parser.add_argument('--elec_type', type = str)
parser.add_argument('--ss_type', type = str)
args = parser.parse_args()

elec_type = args.elec_type
ss_type = args.ss_type

#print(f"elec_type: {elec_type:10}, ss_type: {ss_type}")

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
protein_names = protein_names.iloc[:, 0].tolist()
weight_decay_list = [5e-7, 1e-6, 2e-6]

weight_decay = 1e-6

with open(f"./output/plots/rmsd_and_rg.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    rmsd_md = data['rmsd_md']
    rg_md = data['rg_md']
    rmsd_cg = data['rmsd_cg']
    rg_cg = data['rg_cg']

for name in protein_names:
    rmsd_cg[weight_decay][name] = rmsd_cg[weight_decay][name][10000:]
    rg_cg[weight_decay][name] = rg_cg[weight_decay][name][10000:]    

#### rmsd ####
rmsd_max = {'CLN025': 0.80, '2JOF': 1.28, '1FME': 1.53, '2F4K': 1.53,
         'GTT': 1.53, 'NTL9': 1.30, '2WAV': 2.03, 'PRB': 2.33,
         'UVF': 2.03, 'NuG2': 1.63, 'A3D': 2.53, 'lambda': 1.73}

rg_max = {'CLN025': 0.93, '2JOF': 1.53, '1FME': 2.03, '2F4K': 1.83,
         'GTT': 1.53, 'NTL9': 1.23, '2WAV': 2.03, 'PRB': 2.33,
         'UVF': 2.03, 'NuG2': 1.63, 'A3D': 2.63, 'lambda': 1.73}

stride = 800
fig = plt.figure(figsize = (6.4*2, 4.8*3))

for name in protein_names:
    print(name)
    idx_protein = protein_names.index(name)
    if idx_protein % 2 == 1:
        idx_subplot = idx_protein * 2
    else:
        idx_subplot = idx_protein * 2 + 1
        
    axes = fig.add_subplot(12, 2, idx_subplot)
    plt.plot(rmsd_md[name][::stride], ms = 3, color = 'C1', linewidth = 1)    
    s = int(len(rmsd_cg[weight_decay][name]) / len(rmsd_md[name]) * stride)
    plt.plot(rmsd_cg[weight_decay][name][::s], ms = 3, color = 'C0', linewidth = 1)    
    plt.ylim(0, rmsd_max[name])
    axes.set_xticks([])
    #plt.ylabel("RMSD (nm)")
    
    axes = fig.add_subplot(12, 2, idx_subplot + 2)    
    plt.plot(rg_md[name][::stride], ms = 3, color = 'C1', linewidth = 1)
    s = int(len(rg_cg[weight_decay][name]) / len(rg_md[name]) * stride)
    plt.plot(rg_cg[weight_decay][name][::s], ms = 3, color = 'C0', linewidth = 1)    
    plt.ylim(None, rg_max[name])
    axes.set_xticks([])
    #plt.ylabel("Rg (nm)")    

plt.tight_layout()    
plt.savefig(f"./output/figures/rmsd_rg_traj_all.eps")
plt.savefig(f"./output/figures/rmsd_rg_traj_all.png")    

exit()

ncols = 4
nrows = 3

fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
for name in protein_names:
    print(name)
    axes = fig.add_subplot(nrows, ncols, protein_names.index(name) + 1)
    bins = np.linspace(0, rmsd_max[name], 40)
    axes.hist(rmsd_md[name], bins = bins, alpha = 0.5, label = 'All atom', density = True, color = 'C1', log = True)
    axes.hist(rmsd_cg[weight_decay][name], bins = bins, alpha = 0.5, label = 'CG', density = True, color = 'C0', log = True)
    if name == 'CLN025':
        axes.legend()
    plt.xlim(-0.1, rmsd_max[name])
os.makedirs(f"./output/figures", exist_ok = True)    
plt.savefig(f"./output/figures/rmsd_hist_log.pdf")

exit()

fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
for name in protein_names:
    print(name)
    axes = fig.add_subplot(nrows, ncols, protein_names.index(name) + 1)
    bins = np.linspace(0, rmsd_max[name], 40)
    axes.hist(rmsd_md[name], bins = bins, alpha = 0.5, label = 'All atom', density = True, color = 'C1')
    axes.hist(rmsd_cg[weight_decay][name], bins = bins, alpha = 0.5, label = 'CG', density = True, color = 'C0')
    if name == 'CLN025':
        axes.legend()
    plt.xlim(-0.1, rmsd_max[name])
os.makedirs(f"./output/figures", exist_ok = True)    
plt.savefig(f"./output/figures/rmsd_hist.eps")





stride = 200
fig = plt.figure(figsize = (6.4*4, 4.8*4))
for name in protein_names:
    print(name)
    axes = fig.add_subplot(12, 1, protein_names.index(name) + 1)
    plt.plot(rmsd_md[name][::stride], '.', label = 'All atom', ms = 3, color = 'C1')
    
    s = int(len(rmsd_cg[weight_decay][name]) / len(rmsd_md[name]) * stride)
    
    plt.plot(rmsd_cg[weight_decay][name][::s], '.', label = 'CG', ms = 3, color = 'C0')    
    plt.ylim(0, rmsd_max[name])
    axes.set_xticks([])
    
plt.savefig(f"./output/figures/rmsd_traj.eps")
plt.savefig(f"./output/figures/rmsd_traj.png")    

#### rg ####
ncols = 4
nrows = 3
fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
rg_max = {'CLN025': 0.93, '2JOF': 1.53, '1FME': 2.03, '2F4K': 1.83,
         'GTT': 1.53, 'NTL9': 1.23, '2WAV': 2.03, 'PRB': 2.33,
         'UVF': 2.03, 'NuG2': 1.63, 'A3D': 2.63, 'lambda': 1.73}
for name in protein_names:
    print(name)
    axes = fig.add_subplot(nrows, ncols, protein_names.index(name) + 1)
    rg_min = min(rg_md[name].min(), rg_cg[weight_decay][name].min())
    bins = np.linspace(rg_min, rg_max[name], 40)
    axes.hist(rg_md[name], bins = bins, alpha = 0.5, label = 'All atom', density = True, color = 'C1')
    axes.hist(rg_cg[weight_decay][name], bins = bins, alpha = 0.5, label = 'CG', density = True, color = 'C0')
    if name == 'CLN025':
        axes.legend()
    #plt.xlim(-0.1, rg_max[name])
os.makedirs(f"./output/figures", exist_ok = True)    
plt.savefig(f"./output/figures/rg_hist.eps")

fig = plt.figure(figsize = (6.4*4, 4.8*4))
for name in protein_names:
    print(name)
    axes = fig.add_subplot(12, 1, protein_names.index(name) + 1)
    plt.plot(rg_md[name][::stride], '.', label = 'All atom', ms = 3, color = 'C1')

    s = int(len(rg_cg[weight_decay][name]) / len(rg_md[name]) * stride)
    
    plt.plot(rg_cg[weight_decay][name][::s], '.', label = 'CG', ms = 3, color = 'C0')    
    plt.ylim(None, rg_max[name])
    axes.set_xticks([])
    
plt.savefig(f"./output/figures/rg_traj.eps")
plt.savefig(f"./output/figures/rg_traj.png")

#fig = plt.figure(figsize = (6.4*3, 4.8*3))
fig = plt.figure(figsize = (6.4*3, 4.8*3))
idx_axes = 1
for name in ['2JOF', 'A3D']:
    axes = fig.add_subplot(4, 1, idx_axes)
    plt.plot(rmsd_md[name][::stride], label = 'All atom', ms = 3, color = 'C1', linewidth = 1)
    s = int(len(rmsd_cg[weight_decay][name]) / len(rmsd_md[name]) * stride)
    plt.plot(rmsd_cg[weight_decay][name][::s], label = 'CG', ms = 3, color = 'C0', linewidth = 1)
    plt.ylim(0, rmsd_max[name])
    axes.set_xticks([])
    idx_axes += 1

    axes = fig.add_subplot(4, 1, idx_axes)    
    plt.plot(rg_md[name][::stride], label = 'All atom', ms = 3, color = 'C1', linewidth = 1)
    s = int(len(rg_cg[weight_decay][name]) / len(rg_md[name]) * stride)
    plt.plot(rg_cg[weight_decay][name][::s], label = 'CG', ms = 3, color = 'C0', linewidth = 1)    
    plt.ylim(None, rg_max[name])
    axes.set_xticks([])
    idx_axes += 1
    
plt.savefig(f"./output/figures/rmsd_and_rg_traj.eps")


exit()
