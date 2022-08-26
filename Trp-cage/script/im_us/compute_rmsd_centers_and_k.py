import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
parser.add_argument('--size', type = int, default = 48)
args = parser.parse_args()

name = args.name
size = args.size
print(f"name: {name}, size: {size}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
traj_im = mdtraj.load_dcd(f"./output/{name}/im/CG_simulations/traj.dcd", psf)
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)

rmsd_md = mdtraj.rmsd(traj_md, traj_ref)
rmsd_im = mdtraj.rmsd(traj_im, traj_ref)

rmsd_max = max(rmsd_md.max(), rmsd_im.max())
rmsd_min = 0.0

rmsd_centers = np.linspace(rmsd_min, rmsd_max, size, endpoint = True)
width = (rmsd_max - rmsd_min)/(size - 1)

kbT = 2.4
rmsd_k = 2.4/(width/1.5)**2

exit()

with open(f"./output/{name}/im_us/rmsd_centers_and_k.pkl", 'wb') as file_handle:
    pickle.dump({'centers': rmsd_centers, 'k': rmsd_k}, file_handle)

    
