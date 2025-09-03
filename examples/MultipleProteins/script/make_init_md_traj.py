import mdtraj
import pandas as pd
import numpy as np
from sys import exit
import os
protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
protein_names = protein_names.iloc[:, 0].tolist()

N = 250_000
os.makedirs(f"./data/traj_CG_250K", exist_ok = True)
for name in protein_names:
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")    
    traj = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
    stride = len(traj) // N
    traj = traj[::stride]
    traj.save_dcd(f"./data/traj_CG_250K/{name}.dcd")
    print(f"{name:10}: {len(traj):10,}")

