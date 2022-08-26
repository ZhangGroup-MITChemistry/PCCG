#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=make_system
#SBATCH --time=1:00:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-11
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/make_system_%a.txt

import numpy as np
import simtk.openmm as omm
import simtk.unit as unit
import simtk.openmm.app as ommapp
import math
import mdtraj
import pickle
from sys import exit
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG/")
from CLCG.utils.make_system import *
import argparse
import pandas as pd
import os

protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
job_idx = 1
name = protein_names.iloc[job_idx, 0]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
AA_info = pd.read_csv("./script/md/amino_acids_with_learned_sigmas.csv", index_col = 'name')

masses = [AA_info.loc[r.name, 'mass'] for r in psf.residues]

with open(f"./output/{name}/im/FF/bonded_parameters.pkl", 'rb') as file_handle:
    bonded_parameters = pickle.load(file_handle)

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

protein_info = pd.read_csv('./script/md/protein_temperature_and_ionic_strength.txt', comment = "#", index_col = 0)
T = protein_info.loc[name, 'temperature']
system = make_system(masses, coor_transformer, bonded_parameters, protein_info.loc[name, 'temperature'])

with open(f"./output/{name}/rmsd/rmsd_U.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
rmsd_over_the_range = data['rmsd_over_the_range'],
rmsd_min = data['rmsd_min']
rmsd_max = data['rmsd_max']
U = data['U']

traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)
helix_particle_index = range(2, 14)

Kb = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
Kb = Kb.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)

rmsd_force = omm.RMSDForce(traj_ref.xyz[0], helix_particle_index)
cv_force = omm.CustomCVForce("Kb*T*u_rmsd(rmsd)")
cv_force.addCollectiveVariable('rmsd', rmsd_force)
u_rmsd = omm.Continuous1DFunction(U, rmsd_min, rmsd_max)
cv_force.addTabulatedFunction('u_rmsd', u_rmsd)
cv_force.addGlobalParameter('Kb', Kb)
cv_force.addGlobalParameter('T', T)
system.addForce(cv_force)

xml = omm.XmlSerializer.serialize(system)
f = open(f"./output/{name}/rmsd/CG_system.xml", 'w')
f.write(xml)
f.close()

exit()
