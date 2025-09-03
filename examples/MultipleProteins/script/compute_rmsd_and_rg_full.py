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
protein_names = protein_names.iloc[:, 0].tolist()

psf = {}
traj_ref = {}
for name in protein_names:
    print(name)
    psf[name] = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")    
    traj_ref[name] = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf[name])

print("******* MD *********")    
rmsd_md = {}
rg_md = {}
for name in protein_names:
    print(name)
    traj_md = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", psf[name])
    rmsd_md[name] = mdtraj.rmsd(traj_md, traj_ref[name])
    rg_md[name] = mdtraj.compute_rg(traj_md)

print("******* CG *********")        
rmsd_cg = {}
rg_cg = {}
for weight_decay in weight_decay_list:
    rmsd_cg[weight_decay] = {}
    rg_cg[weight_decay] = {}    
    for name in protein_names:
        print(weight_decay, name)
        traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.dcd",
                                  psf[name], stride = 4)    
        rmsd_cg[weight_decay][name] = mdtraj.rmsd(traj_cg, traj_ref[name])
        rg_cg[weight_decay][name] = mdtraj.compute_rg(traj_cg)

with open(f"./output/plots/rmsd_and_rg.pkl", 'wb') as file_handle:
    pickle.dump({'rmsd_md': rmsd_md,
                 'rg_md': rg_md,
                 'rmsd_cg': rmsd_cg,
                 'rg_cg': rg_cg}, file_handle)
    
exit()
