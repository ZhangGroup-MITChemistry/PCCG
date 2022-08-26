import mdtraj
import numpy as np
import pickle
import pandas as pd
from itertools import product
from sys import exit

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
weight_decay_list = [1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6]
protein_names = protein_names.iloc[0:3, 0].tolist()

for name, weight_decay in product(protein_names, weight_decay_list):
    print(f"name: {name}, weight_decay: {weight_decay}", flush = True)
    
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    with open(f"./output/{name}/TRE/weight_decay_{weight_decay:.3E}/log_temperatures.pkl", 'rb') as file_handle:
        log_t = pickle.load(file_handle)

    size = log_t.shape[0]
    T = log_t.min()
    traj_list = []
    for rank in range(size):
        traj = mdtraj.load_dcd(f"./output/{name}/TRE/weight_decay_{weight_decay:.3E}/traj_{rank}.dcd", psf)
        traj = traj[log_t[rank] == T]
        traj_list.append(traj)
    traj = mdtraj.join(traj_list)
    traj.save_dcd(f"./output/{name}/TRE/weight_decay_{weight_decay:.3E}/traj.dcd")
    
