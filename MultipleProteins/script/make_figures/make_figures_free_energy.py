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

protein_names_alt = {
    'CLN025': 'Chignolin',
    '2JOF': 'Trp-cage',
    '1FME': 'BBA',
    '2F4K': 'Villin',
    'GTT': 'WW domain',
    'NTL9': 'NTL9',
    '2WAV': 'BBL',
    'PRB': 'Protein B',
    'UVF': 'Homeodomain',
    'NuG2': 'Protein G',
    'A3D': '$\alpha$3D',
    'lambda': '$\lambda$-repressor'
}

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
    
ncols = 4
nrows = 3

fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
fig.clf()
for name in protein_names:
    print(name)
    axes = fig.add_subplot(nrows, ncols, protein_names.index(name) + 1)
    bins = np.linspace(0, rmsd_max[name], 40)
    density, bin_edges = np.histogram(rmsd_md[name], bins = bins, density = True)
    F_md = -np.log(density)

    density, bin_edges = np.histogram(rmsd_cg[weight_decay][name], bins = bins, density = True)
    F_cg = -np.log(density)
    F_cg = F_cg - F_cg.min() + F_md.min()
    
    plt.plot((bins[0:-1]+bins[1:])/2., F_md, color = 'C1', label = 'All atom', linewidth = 3.0)
    plt.plot((bins[0:-1]+bins[1:])/2., F_cg, color = 'C0', label = 'CG', linewidth = 3.0)    
    axes.legend()
    if name == 'A3D':
        plt.xlabel("RMSD (nm)"
                   "\n"
                   r"$\alpha$-3D")
    elif name == 'lambda':
        plt.xlabel("RMSD (nm)"
                   "\n"
                   r"$\lambda$-repressor")
    else:
        xlabel = f'RMSD (nm) \n {protein_names_alt[name]}'
        plt.xlabel(xlabel)
        
    plt.ylabel(r'Free energy ($k_B$T)')    
    # axes.hist(rmsd_md[name], bins = bins, alpha = 0.5, label = 'All atom', density = True, color = 'C1', log = True)
    # axes.hist(rmsd_cg[weight_decay][name], bins = bins, alpha = 0.5, label = 'CG', density = True, color = 'C0', log = True)
    # if name == 'CLN025':
    #     axes.legend()
    # plt.xlim(-0.1, rmsd_max[name])

os.makedirs(f"./output/figures", exist_ok = True)
#plt.tight_layout()
plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
plt.savefig(f"./output/figures/rmsd_free_energy.eps")


fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
fig.clf()
for name in protein_names:
    print(name)
    axes = fig.add_subplot(nrows, ncols, protein_names.index(name) + 1)
    bins = np.linspace(0, rg_max[name], 40)
    density, bin_edges = np.histogram(rg_md[name], bins = bins, density = True)
    F_md = -np.log(density)

    density, bin_edges = np.histogram(rg_cg[weight_decay][name], bins = bins, density = True)
    F_cg = -np.log(density)
    F_cg = F_cg - F_cg.min() + F_md.min()
    
    plt.plot((bins[0:-1]+bins[1:])/2., F_md, color = 'C1', label = 'All atom', linewidth = 3.0)
    plt.plot((bins[0:-1]+bins[1:])/2., F_cg, color = 'C0', label = 'CG', linewidth = 3.0)    
    axes.legend()
    if name == 'A3D':
        plt.xlabel("Rg (nm)"
                   "\n"
                   r"$\alpha$-3D")
    elif name == 'lambda':
        plt.xlabel("Rg (nm)"
                   "\n"
                   r"$\lambda$-repressor")
    else:
        xlabel = f'Rg (nm) \n {protein_names_alt[name]}'
        plt.xlabel(xlabel)
        
    plt.ylabel(r'Free energy ($k_B$T)')    
    # axes.hist(rg_md[name], bins = bins, alpha = 0.5, label = 'All atom', density = True, color = 'C1', log = True)
    # axes.hist(rg_cg[weight_decay][name], bins = bins, alpha = 0.5, label = 'CG', density = True, color = 'C0', log = True)
    # if name == 'CLN025':
    #     axes.legend()
    # plt.xlim(-0.1, rg_max[name])

os.makedirs(f"./output/figures", exist_ok = True)
#plt.tight_layout()
plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
plt.savefig(f"./output/figures/rg_free_energy.eps")

exit()
