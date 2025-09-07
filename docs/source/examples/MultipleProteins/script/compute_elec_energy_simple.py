#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=elec
#SBATCH --time=03:00:00
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --array=0-11
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/elec_%a.txt

import numpy as np
import simtk.openmm.app  as app
import simtk.openmm as omm
import simtk.unit as unit
import argparse
import mdtraj
from sys import exit
import time
import pickle
import pandas as pd
import os
import torch

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")

traj_md = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", psf, stride = 1)
with open(f"./output/{name}/rmsd_centers_and_k_for_imus.pkl", 'rb') as file_handle:
    rmsd_centers_and_k = pickle.load(file_handle)
size = rmsd_centers_and_k['size']
traj_imus = []
for rank in range(size):
    print(f"rank: {rank}")
    traj = mdtraj.load_dcd(f"./output/{name}/traj_imus/traj_{rank}.dcd", psf)
    traj_imus.append(traj)
traj_imus = mdtraj.join(traj_imus)
stride = size // 10
traj_imus = traj_imus[::stride]

with open(f"./output/{name}/system_im.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

protein_info = pd.read_csv("./info/protein_temperature_and_ionic_strength.txt", index_col = 'name', comment = '#')
AA_info = pd.read_csv("./info/amino_acids_with_learned_sigmas.csv", index_col = 'name')

## make an integrator
T = protein_info.loc[name, 'temperature']
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.kelvin * unit.AVOGADRO_CONSTANT_NA

def make_custom_elec_force():
    formula = ["ONE_4PI_EPS0*charge1*charge2/(r*epsilon)",
               "ONE_4PI_EPS0 = 138.935456",
               "epsilon = 78.4"
    ]
    custom_elec_force = omm.CustomNonbondedForce(";".join(formula))
    custom_elec_force.addPerParticleParameter('charge')
    return custom_elec_force

custom_elec_force = make_custom_elec_force()
custom_elec_force.setForceGroup(1)
for i in range(psf.n_residues):
    resname = psf.residue(i).name
    charge = AA_info.loc[resname, 'charge']
    custom_elec_force.addParticle([charge])
bonds = [[i, i+1] for i in range(psf.n_residues - 1)]
custom_elec_force.createExclusionsFromBonds(bonds, 3)    
system.addForce(custom_elec_force)

fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)
platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)

xyz_md = traj_md.xyz
elec_energy_md = []
for i in range(xyz_md.shape[0]):
    context.setPositions(xyz_md[i])
    state = context.getState(getEnergy = True, groups = set([1]))
    U = state.getPotentialEnergy()
    elec_energy_md.append(-U/kbT)
    
    if (i+1) % 10000 == 0:
        print(i)

xyz_imus = traj_imus.xyz
elec_energy_imus = []
for i in range(xyz_imus.shape[0]):
    context.setPositions(xyz_imus[i])
    state = context.getState(getEnergy = True, groups = set([1]))
    U = state.getPotentialEnergy()
    elec_energy_imus.append(-U/kbT)
    if (i+1) % 10000 == 0:
        print(i)
        
elec_energy_md = torch.tensor(elec_energy_md)
elec_energy_imus = torch.tensor(elec_energy_imus)

xml = omm.XmlSerializer.serialize(system)
f = open(f"./output/{name}/system_im_with_elec_simple.xml", 'w')
f.write(xml)
f.close()

os.makedirs(f"./output/{name}", exist_ok = True)
torch.save({'md': elec_energy_md,
            'imus': elec_energy_imus},
           f"./output/{name}/elec_energy_simple.pt")
exit()

# ONE_4PI_EPS0 = 138.935456
# EPS0 = 1./(ONE_4PI_EPS0*4*math.pi)
# kbT = unit.BOLTZMANN_CONSTANT_kB*300*unit.kelvin*unit.AVOGADRO_CONSTANT_NA
# kbT = kbT.value_in_unit(unit.kilojoule_per_mole)
# ionic_strength = 2/(40*unit.angstroms)**3
# ionic_strength = ionic_strength.value_in_unit(unit.nanometer**(-3))
# lambda_D = math.sqrt(EPS0*kbT/(2*ionic_strength))

# ionic_strength = protein_info.loc[name, 'ionic_strength']

# def make_custom_elec_force():
#     formula = ["ONE_4PI_EPS0*charge1*charge2/(r*epsilon)*exp(-r/lambda_D)",
#                "lambda_D = sqrt(EPS0*epsilon*kb*Temperature/(2*ionic_strength))",
#                "epsilon = A + B/(1 + kappa*exp(-lambda*B*r))",
#                "EPS0 = 1./(ONE_4PI_EPS0*4*PI)",
#                "PI = 3.141592653",
#                "B = 78.4 - A; A = -8.5525; kappa = 7.7839; lambda = 0.03627",
#                "kb = 2.4943387854459713/300",
#                "ONE_4PI_EPS0 = 138.935456"
#     ]

#     custom_elec_force = omm.CustomNonbondedForce(";".join(formula))
#     custom_elec_force.addGlobalParameter("Temperature", 300)
#     ionic_strength = 150*unit.millimolar
#     ionic_strength = ionic_strength*unit.AVOGADRO_CONSTANT_NA
#     ionic_strength = ionic_strength.value_in_unit(unit.nanometer**(-3))
#     custom_elec_force.addGlobalParameter("ionic_strength", ionic_strength)
#     custom_elec_force.addPerParticleParameter('charge')
#     print("remember to set values for global parameters T and ionic_strength")
#     return custom_elec_force
