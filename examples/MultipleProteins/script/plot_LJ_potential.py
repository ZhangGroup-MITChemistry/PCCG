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

with open(f"./output/common/LJ_rmin.pkl", 'rb') as file_handle:
    r_min_dict = pickle.load(file_handle)
aa_pairs = list(r_min_dict.keys())
aa_pairs.sort()

AA_info = pd.read_csv("./info/amino_acids_with_learned_sigmas.csv",
                      index_col = 'name')
AA_names = list(AA_info.index)

AA_name_dict = {'G': 'GLY', 'P': 'PRO', 'A': 'ALA', 'V': 'VAL',
                'L': 'LEU', 'I': 'ILE', 'M': 'MET', 'C': 'CYS',
                'F': 'PHE', 'Y': 'TYR', 'W': 'TRP', 'H': 'HIS',
                'K': 'LYS', 'R': 'ARG', 'Q': 'GLN', 'N': 'ASN',
                'E': 'GLU', 'D': 'ASP', 'S': 'SER', 'T': 'THR'}

NUM_COLORS = 20

name = "CLN025"

weight_decay_list = [1e-10, 1e-9, 1e-8,
                     1e-7, 1e-6, 1e-5,
                     1e-4, 1e-3]
weight_decay = 1e-6

pdf = PdfPages(f"./output/plots/LJ_potential.pdf")

for weight_decay in weight_decay_list:
    print(weight_decay)
    
    data = torch.load(f"./output/{name}/FF/FF_weight_decay_{weight_decay:.3E}.pt",
                      map_location=torch.device('cpu'))
    FF = data['FF']
    bonded_ff, lj_ff, rmsd_ff = FF['bonded'], FF['lj'], FF['rmsd']

    ncols = 5
    nrows = 4
    fig = plt.figure(figsize = (6.4*ncols, 4.8*nrows))
    fig.suptitle(f'LJ_weight_decay:{weight_decay:.3E}')
    for aa1 in AA_names:
        print(aa1)
        plt.subplot(nrows, ncols, AA_names.index(aa1) + 1)
        colors = iter([cm.tab20(i) for i in range(20)])
        for aa2 in AA_names:
            aa = [aa1, aa2]
            aa.sort()
            if (aa[0], aa[1]) in aa_pairs:
                idx = aa_pairs.index((aa[0], aa[1]))
                r = np.linspace(lj_ff['min'], lj_ff['max'], lj_ff['U'].shape[-1])
                plt.plot()
                plt.plot(r, lj_ff['U'][idx],
                        linewidth = 2.0,
                        label = aa2,
                        color = next(colors))
        plt.ylim(-10, 10)
        plt.xlim(0, lj_ff['max'])
        #plt.legend(ncol = 2, loc = 'right')
        plt.title(aa1)
    pdf.savefig()

pdf.close()

exit()

ncol = 6
nrow = len(aa_pairs)//ncol + 1

fig = plt.figure(np.random.randint(1000), figsize = (6.4*ncol, 4.8*nrow))
fig.clf()
for k in range(len(aa_pairs)):
    print(k)
    plt.subplot(nrow,ncol,k+1)
    plt.plot(LJ_parameters[aa_pairs[k]]['r_over_range'], LJ_parameters[aa_pairs[k]]['U'], linewidth = 4.0)
    plt.ylim(-10, 10)
    plt.xlim(0, 1.5)
    aa1, aa2 = aa_pairs[k]
    plt.title(f"{aa1}-{aa2}")
plt.tight_layout()    
os.makedirs("./output/plots", exist_ok = True)
plt.savefig(f"./output/plots/LJ_potential_split_weight_decay_{weight_decay:.3E}.pdf")

AA_names = [AA_name_dict[n] for n in 'RHKDESTNQCGPAVILMFYW']
        
E_min = np.zeros((len(AA_names), len(AA_names)))
for i in range(len(AA_names)):
    for j in range(len(AA_names)):
        aa1, aa2 = AA_names[i], AA_names[j]
        aa = [aa1, aa2]
        aa.sort()
        if (aa[0], aa[1]) in aa_pairs:
            E_min[i,j] = np.min(LJ_parameters[(aa[0], aa[1])]['U'])
        else:
            E_min[i,j] = None

fig = plt.figure(np.random.randint(1000), figsize = (6.4*2, 4.8*2))
fig.clf()
ax = plt.subplot()
im, cbar = heatmap(E_min, AA_names, AA_names, ax = ax, origin = 'lower')
fig.tight_layout()
plt.savefig(f"./output/plots/LJ_heatmap_weight_decay_{weight_decay:.3E}.pdf")

AA_names = [AA_name_dict[n] for n in 'MRGDPNKSAFVLTYHEQIWC']
E_min = np.zeros((len(AA_names), len(AA_names)))
for i in range(len(AA_names)):
    for j in range(len(AA_names)):
        aa1, aa2 = AA_names[i], AA_names[j]
        aa = [aa1, aa2]
        aa.sort()
        if (aa[0], aa[1]) in aa_pairs:
            E_min[i,j] = np.min(LJ_parameters[(aa[0], aa[1])]['U'])
        else:
            E_min[i,j] = None

fig = plt.figure(np.random.randint(1000), figsize = (6.4*2, 4.8*2))
fig.clf()
ax = plt.subplot()
im, cbar = heatmap(E_min, AA_names, AA_names, ax = ax)
fig.tight_layout()
plt.savefig(f"./output/plots/LJ_heatmap_MOFF_weight_decay_{weight_decay:.3E}.pdf")

exit()


# fig = plt.figure(np.random.randint(1000), figsize = (6.4*2, 4.8*2))
# ax = plt.subplot()
# im = ax.imshow(E_min)
# ax.set_xticks(np.arange(E_min.shape[0]))
# ax.set_yticks(np.arange(E_min.shape[1]))
# ax.set_xticklabels(AA_names)
# ax.set_yticklabels(AA_names)
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
# plt.legend()
# fig.tight_layout()
# plt.savefig(f"./output/plots/LJ_heatmap_weight_decay_{weight_decay:.3E}.pdf")

exit()



ncol = 6
nrow = len(aa_pairs)//ncol + 1

fig = plt.figure(np.random.randint(1000), figsize = (6.4*ncol, 4.8*nrow))
fig.clf()
for k in range(len(aa_pairs)):
    print(k)
    plt.subplot(nrow,ncol,k+1)
    plt.plot(LJ_parameters[aa_pairs[k]]['r_over_range'], LJ_parameters[aa_pairs[k]]['U'], linewidth = 2.0)
    plt.ylim(-10, 20)

os.makedirs("./output/plots", exist_ok = True)
plt.savefig(f"./output/plots/LJ_potential_weight_decay_{weight_decay:.3E}.pdf")

