import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import exit
import pandas as pd
import openmm.unit as unit

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
protein_info = pd.read_csv("./info/protein_temperature_and_ionic_strength.txt",
                           index_col = 'name', comment = '#')

for name in protein_names.iloc[0:2, 0]:
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
    traj_im = mdtraj.load_dcd(f"./output/{name}/traj_im/traj.dcd", psf)
    traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)

    rmsd_md = mdtraj.rmsd(traj_md, traj_ref)
    rmsd_im = mdtraj.rmsd(traj_im, traj_ref)

    rmsd_max = max(rmsd_md.max(), rmsd_im.max())
    rmsd_max = min(2.0, rmsd_max)
    rmsd_min = 0.0

    width = 0.03
    size = int((rmsd_max - rmsd_min)/width) + 1
    
    rmsd_centers = np.linspace(rmsd_min, rmsd_max, size, endpoint = True)

    T = protein_info.loc[name, 'temperature'] * unit.kelvin
    kbT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*T
    kbT = kbT.value_in_unit(unit.kilojoule_per_mole)
    rmsd_k = 2.4/(width/1.5)**2

    print(f"name: {name}, rmsd_k: {rmsd_k:.2f}, size: {size}")
    
    with open(f"./output/{name}/rmsd_centers_and_k_for_imus.pkl", 'wb') as file_handle:
        pickle.dump({'size': size, 'centers': rmsd_centers, 'k': rmsd_k}, file_handle)
