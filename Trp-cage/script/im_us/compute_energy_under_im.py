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
import openmm as omm

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
args = parser.parse_args()

name = args.name
print(f"name: {name}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

with open(f"./output/{name}/im_us/rmsd_centers_and_k.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
rmsd_centers = data['centers']
rmsd_k = data['k']

with open(f"./output/{name}/im/FF/CG_system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

protein_info = pd.read_csv("./script/md/protein_temperature_and_ionic_strength.txt", index_col = 'name', comment = '#')

## make an integrator
T = protein_info.loc[name, 'temperature']
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.kelvin * unit.AVOGADRO_CONSTANT_NA

fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)
platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)

size = len(rmsd_centers)
energy = []

for rank in range(size):
    traj = mdtraj.load_dcd(f"./output/{name}/im_us/trajs/traj_{rank}.dcd", psf, stride = 1)
    energy.append([])    
    for i in range(len(traj)):
        context.setPositions(traj.xyz[i])
        state = context.getState(getEnergy = True)
        U = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energy[-1].append(U)
    print(rank)
energy_us = np.array(energy)


traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf, stride = 1)
energy_md = []
for i in range(len(traj_md)):
    context.setPositions(traj_md.xyz[i])
    state = context.getState(getEnergy = True)
    U = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energy_md.append(U)
energy_md = np.array(energy_md)

with open(f"./output/{name}/im_us/energy_under_im.pkl", 'wb') as file_handle:
    pickle.dump({'md': energy_md, 'us': energy_us}, file_handle)
    

