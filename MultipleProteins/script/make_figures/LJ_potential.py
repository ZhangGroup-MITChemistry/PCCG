import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
import pandas as pd
from sys import exit
from matplotlib.pyplot import cm
import argparse
import sys
import torch
from functions import *
import scipy.cluster as cluster

with open(f"./output/common/LJ_rmin.pkl", 'rb') as file_handle:
    r_min_dict = pickle.load(file_handle)
aa_pairs = list(r_min_dict.keys())
aa_pairs.sort()

AA_info = pd.read_csv("./info/amino_acids_with_learned_sigmas.csv",
                      index_col = 'name')
AA_names = list(AA_info.index)
AA_names.pop(AA_names.index('CYS'))
# AA_name_dict = {'G': 'GLY', 'P': 'PRO', 'A': 'ALA', 'V': 'VAL',
#                 'L': 'LEU', 'I': 'ILE', 'M': 'MET', 'C': 'CYS',
#                 'F': 'PHE', 'Y': 'TYR', 'W': 'TRP', 'H': 'HIS',
#                 'K': 'LYS', 'R': 'ARG', 'Q': 'GLN', 'N': 'ASN',
#                 'E': 'GLU', 'D': 'ASP', 'S': 'SER', 'T': 'THR'}

AA_name_dict = {'G': 'GLY', 'P': 'PRO', 'A': 'ALA', 'V': 'VAL',
                'L': 'LEU', 'I': 'ILE', 'M': 'MET',
                'F': 'PHE', 'Y': 'TYR', 'W': 'TRP', 'H': 'HIS',
                'K': 'LYS', 'R': 'ARG', 'Q': 'GLN', 'N': 'ASN',
                'E': 'GLU', 'D': 'ASP', 'S': 'SER', 'T': 'THR'}
NUM_COLORS = 20

name = "CLN025"
weight_decay = 1e-6

elec_type = 'DH_2'
ss_type = 'simple'

data = torch.load(f"./output/{name}/FF/FF_elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.pt",
                  map_location=torch.device('cpu'))

# data = torch.load(f"./output/{name}/FF_bound_0/FF_elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.pt",
#                   map_location=torch.device('cpu'))

FF = data['FF']
bonded_ff, lj_ff, rmsd_ff = FF['bonded'], FF['lj'], FF['rmsd']

ncols = 4
nrows = 5
fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
for aa1 in AA_names:
    print(aa1)
    ax = plt.subplot(nrows, ncols, AA_names.index(aa1) + 1)
    colors = iter([cm.tab20(i) for i in range(20)])

    lines = []
    for aa2 in AA_names:
        aa = [aa1, aa2]
        aa.sort()
        if (aa[0], aa[1]) in aa_pairs:
            idx = aa_pairs.index((aa[0], aa[1]))
            r = np.linspace(lj_ff['min'], lj_ff['max'], lj_ff['U'].shape[-1])
            line = ax.plot(r, lj_ff['U'][idx],
                           linewidth = 2.0,
                           label = aa2,
                           color = next(colors))
            lines.append(line)
    plt.ylim(-4.5, 4.5)
    plt.xlim(0, lj_ff['max'])
    #plt.title(aa1)
    plt.text(0.1, 3, aa1)
    #plt.xlabel('distance (nm)')
    
ax = plt.subplot(nrows, ncols, 20)
colors = iter([cm.tab20(i) for i in range(20)])
#ax.legend(lines, AA_names)
for aa2 in AA_names:
    plt.plot([], [],
            linewidth = 2.0,
            label = aa2,
            color = next(colors))
ax.legend(ncol = 3)
ax.axis('off')

#plt.tight_layout()
plt.savefig("./output/figures/lj_potential.pdf")    

# pair_names = [['TRP', 'TYR'],
#               ['THR', 'TYR'],
#               ['MET', 'SER'],
#               ['PRO', 'TRP'],
#               ['PRO', 'THR']]

pair_names = [['ALA', 'LYS'],
              ['ILE', 'VAL'],
              ['TRP', 'TYR'],
              ['ASP', 'THR']]

fig = plt.figure()
fig.clf()
for name in pair_names:
    name.sort()
    idx = aa_pairs.index(tuple(name))
    r = np.linspace(lj_ff['min'], lj_ff['max'], lj_ff['U'].shape[-1])
    plt.plot(r, lj_ff['U'][idx], linewidth = 2.0, label = '-'.join(name))
    plt.ylim(-4.5, 4.5)
plt.legend(ncol = 2)
plt.xlabel('Distance (nm)')
plt.ylabel('Energy (kJ/mol)')
plt.tight_layout()
plt.savefig("./output/figures/lj_potential_selected.eps")    

exit()


# pdf = PdfPages("./output/figures/lj_potential_seperate.pdf")
# for aa1 in AA_names:
#     print(aa1)
#     ncols = 5
#     nrows = 4

#     fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
#     fig.suptitle(aa1)
    
#     for aa2 in AA_names:
#         plt.subplot(nrows, ncols, AA_names.index(aa2) + 1)
#         colors = iter([cm.tab20(i) for i in range(20)])
#         aa = [aa1, aa2]
#         aa.sort()
#         print(aa)
#         if (aa[0], aa[1]) in aa_pairs:
#             idx = aa_pairs.index((aa[0], aa[1]))
#             r = np.linspace(lj_ff['min'], lj_ff['max'], lj_ff['U'].shape[-1])
#             plt.plot(r, lj_ff['U'][idx],
#                     linewidth = 3.0,
#                     label = aa2,
#                     color = next(colors))
#             plt.legend()
            
#         plt.ylim(-4.5, 4.5)
#         plt.xlim(0, lj_ff['max'])
#     pdf.savefig()
# pdf.close()

aa_names = 'RHKDESTNQGPAVILMFYW'
AA_names = [AA_name_dict[n] for n in aa_names]
        
E_min = np.zeros((len(AA_names), len(AA_names)))
for i in range(len(AA_names)):
    for j in range(len(AA_names)):
        aa1, aa2 = AA_names[i], AA_names[j]
        aa = [aa1, aa2]
        aa.sort()
        if (aa[0], aa[1]) in aa_pairs:
            idx = aa_pairs.index((aa[0], aa[1]))
            E_min[i,j] = lj_ff['U'][idx].min().item()
        elif (aa[0], aa[1]) == ('TRP', 'TRP'):
            idx = aa_pairs.index(('TRP', 'TYR'))
            E_min[i,j] = lj_ff['U'][idx].min().item()
        else:
            E_min[i,j] = None

fig = plt.figure(figsize = (6.4*2, 4.8*2))
fig.clf()
ax = plt.subplot()
im, cbar = heatmap(E_min, AA_names, AA_names, ax = ax, origin = 'lower')
fig.tight_layout()
plt.savefig(f"./output/figures/lj_heatmap.pdf")

exit()


linkage = cluster.hierarchy.linkage(E_min)


exit()

AA_names = [AA_name_dict[n] for n in 'MRGDPNKSAFVLTYHEQIWC']
E_min = np.zeros((len(AA_names), len(AA_names)))
for i in range(len(AA_names)):
    for j in range(len(AA_names)):
        aa1, aa2 = AA_names[i], AA_names[j]
        aa = [aa1, aa2]
        aa.sort()
        if (aa[0], aa[1]) in aa_pairs:
            idx = aa_pairs.index((aa[0], aa[1]))
            E_min[i,j] = lj_ff['U'][idx].min().item()
        else:
            E_min[i,j] = None

fig = plt.figure(figsize = (6.4*2, 4.8*2))
fig.clf()
ax = plt.subplot()
im, cbar = heatmap(E_min, AA_names, AA_names, ax = ax, origin = 'lower')
fig.tight_layout()
plt.savefig(f"./output/figures/lj_heatmap_MOFF_order.pdf")

exit()
