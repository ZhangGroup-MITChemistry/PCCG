import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad
import pickle
import mdtraj
from sys import exit
import os
import argparse
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
from CLCG.utils.splines import *
from CLCG.utils.CL import *
import torch
torch.set_default_dtype(torch.double)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import pandas as pd
from collections import defaultdict

protein_names = pd.read_csv("./info/protein_names.txt",
                            comment = "#",
                            header = None)

# job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
# job_idx = 0
# name = protein_names.iloc[job_idx, 0]

aa_info = pd.read_csv("./info/amino_acids_with_learned_sigmas.csv")
aa_names = list(aa_info['name'])
aa_names.sort()
aa_pairs = []
for i in range(len(aa_names)):
    for j in range(i, len(aa_names)):
        aa_pairs.append((aa_names[i], aa_names[j]))
num_aa_pairs = len(aa_pairs)
r_min_dict = defaultdict(list)

def sort_pair_name(pair):
    if pair[0] <= pair[1]:
        return pair
    else:
        tmp = pair[0]
        pair[0] = pair[1]
        pair[1] = tmp
        return pair

for name in list(protein_names.iloc[:,0]):
    print(name)
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    resnames = [res.name for res in psf.residues]

    traj_md = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", psf, stride = 1)
    indices = [(i,j) for i in range(psf.n_residues) for j in range(i+4, psf.n_residues)]
    r = mdtraj.compute_distances(traj_md, indices)
    r_min = r.min(0)

    for k in range(len(indices)):
        i,j = indices[k]
        pair = sort_pair_name([resnames[i], resnames[j]])
        r_min_dict[tuple(pair)].append(r_min[k])

for key in r_min_dict.keys():
    r_min_dict[key] = np.min(r_min_dict[key])
r_min_dict = dict(r_min_dict)

os.makedirs(f"./output/common", exist_ok = True)    
with open(f"./output/common/LJ_rmin.pkl", 'wb') as file_handle:
    pickle.dump(r_min_dict, file_handle)
    
