import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
from FastMBAR import *
import openmm.unit as unit
import pandas  as pd
from sys import exit
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
args = parser.parse_args()

name = args.name
print(f"name: {name}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)

with open(f"./output/{name}/im_us/rmsd_centers_and_k.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
rmsd_centers = data['centers']
rmsd_k = data['k']

size = len(rmsd_centers)
rmsd = []

for rank in range(size):
    traj = mdtraj.load_dcd(f"./output/{name}/im_us/trajs/traj_{rank}.dcd", psf, stride = 1)
    rmsd.append(mdtraj.rmsd(traj, traj_ref))
    print(rank)
    
num_frames = np.array([len(rmsd[rank]) for rank in range(size)])
rmsd = np.concatenate(rmsd)

energy_matrix = 0.5*rmsd_k*(rmsd[np.newaxis, :] - rmsd_centers[:, np.newaxis])**2

with open(f"./output/{name}/im_us/energy_under_im.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    energy_us_under_im = data['us']
    energy_md_under_im = data['md']    
    
energy_matrix = energy_matrix + energy_us_under_im.reshape(-1)

protein_info = pd.read_csv("./script/md/protein_temperature_and_ionic_strength.txt", index_col = 'name', comment = '#')
T = protein_info.loc[name, 'temperature']
Kb = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
Kb = Kb.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
KbT = Kb*T
energy_matrix = energy_matrix / KbT
fastmbar = FastMBAR(energy_matrix, num_frames, verbose = True, cuda = True)
log_q_us = fastmbar.log_prob_mix

traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf, stride = 1)
rmsd_md = mdtraj.rmsd(traj_md, traj_ref)

energy_matrix_md = 0.5*rmsd_k*(rmsd_md[np.newaxis, :] - rmsd_centers[:, np.newaxis])**2
energy_matrix_md = energy_matrix_md + energy_md_under_im
energy_matrix_md = energy_matrix_md / KbT

biased_energy = energy_matrix_md + fastmbar.bias_energy.reshape((-1,1))
biased_energy_min = np.min(biased_energy, 0, keepdims = True)        
log_q_md = np.log(np.sum(np.exp(-(biased_energy - biased_energy_min)), 0)) - biased_energy_min.reshape(-1)

with open(f"./output/{name}/im_us/log_q.pkl", 'wb') as file_handle:
    pickle.dump({'log_q_md': log_q_md,
                 'log_q_us': log_q_us},
                file_handle)
exit()
