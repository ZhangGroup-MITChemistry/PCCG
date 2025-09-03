import mdtraj
import argparse
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sys import exit
from itertools import product
import pandas as pd
import ray
import os
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

parser = argparse.ArgumentParser()
parser.add_argument('--elec_type', type = str)
parser.add_argument('--ss_type', type = str)
args = parser.parse_args()

elec_type = args.elec_type
ss_type = args.ss_type

print(f"elec_type: {elec_type:10}, ss_type: {ss_type}")

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
weight_decay_list = [5e-7,
                     1e-6, 2e-6]
#protein_names = protein_names.iloc[[1,-2], 0].tolist()
protein_names = protein_names.iloc[:, 0].tolist()

psf = {}
traj_ref = {}
for name in protein_names:
    print(name)
    psf[name] = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")    
    traj_ref[name] = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf[name])
    traj_ref[name].save_xyz(f"./output/figures/structure_with_lowest_rmsd/{name}_reference.xyz")

print("******* CG *********")        
rmsd_cg = {}
weight_decay = 1e-6

for name in protein_names:
    traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.dcd",
                              psf[name], stride = 4)
    traj_cg = traj_cg[10000:]

    N = 5_000
    stride = int(traj_cg.n_frames / N)
    traj_sub = traj_cg[::stride]
    traj_sub = traj_sub[0:N]

    distances = np.empty((traj_sub.n_frames, traj_sub.n_frames))
    for i in range(traj_sub.n_frames):
        distances[i] = mdtraj.rmsd(traj_sub, traj_sub, i)    
        # if (i + 1) % 200 == 0:
        #     print(i)

    #print('Max pairwise rmsd: %f nm' % np.max(distances))
    reduced_distances = squareform(distances, checks=False)

    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')
    n_cluster = 50
    cluster_label = scipy.cluster.hierarchy.fcluster(linkage, t = n_cluster, criterion = 'maxclust')
    cluster_size = []
    for i in range(1, n_cluster+1):
        cluster_size.append(np.sum(cluster_label == i))

    popular_cluster_label = np.argmax(cluster_size) + 1
    flag = cluster_label == popular_cluster_label
    traj_sub = traj_sub[flag]

    dist = np.mean(distances[flag,:][:,flag], -1)
    idx = np.argmin(dist)

    os.makedirs(f"./output/figures/structure_cluster/", exist_ok = True)
    traj_sub.superpose(traj_ref[name])
    traj_sub[idx].save_xyz(f"./output/figures/structure_with_lowest_rmsd/{name}.xyz")

    rmsd = mdtraj.rmsd(traj_sub[idx], traj_ref[name])
    print(weight_decay, name, rmsd)
    
exit()
