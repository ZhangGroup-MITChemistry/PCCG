#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=make_reference_structure
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-11
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/make_reference_%a.txt

import mdtraj
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
import pandas as pd
import os
from sys import exit

protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 1
name = protein_names.iloc[job_idx, 0]

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type = str)
# args = parser.parse_args()
# model = args.model

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf, stride = 1)

N = 5_000
stride = int(traj.n_frames / N)
traj_sub = traj[::stride]
traj_sub = traj_sub[0:N]

distances = np.empty((traj_sub.n_frames, traj_sub.n_frames))
for i in range(traj_sub.n_frames):
    print(i)
    distances[i] = mdtraj.rmsd(traj_sub, traj_sub, i)
print('Max pairwise rmsd: %f nm' % np.max(distances))
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

traj_sub[idx].save_xyz(f"./output/{name}/md/reference_structure.xyz")

