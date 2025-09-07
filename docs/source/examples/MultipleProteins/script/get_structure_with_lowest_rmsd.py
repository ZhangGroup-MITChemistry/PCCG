import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
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
    rmsd_cg[name] = mdtraj.rmsd(traj_cg, traj_ref[name])

    idx = rmsd_cg[name].argmin()
    os.makedirs(f"./output/figures/structure_with_lowest_rmsd/", exist_ok = True)
    traj_cg.superpose(traj_ref[name])
    
    traj_cg[idx].save_xyz(f"./output/figures/structure_with_lowest_rmsd/{name}.xyz")

    rmsd = mdtraj.rmsd(traj_cg[idx], traj_ref[name])
    print(weight_decay, name, rmsd)
    
exit()
