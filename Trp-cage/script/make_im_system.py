#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=make_system
#SBATCH --time=1:00:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-1
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

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
name = protein_names.iloc[job_idx, 0]

protein_info = pd.read_csv("./info/protein_temperature_and_ionic_strength.txt",
                           index_col = 'name', comment = '#')
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
AA_info = pd.read_csv("./info/amino_acids_with_learned_sigmas.csv", index_col = 'name')
masses = [AA_info.loc[r.name, 'mass'] for r in psf.residues]

with open(f"./output/{name}/bonded_parameters_im.pkl", 'rb') as file_handle:
    bonded_parameters = pickle.load(file_handle)

with open(f"./output/{name}/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

system = make_system(masses,
                     coor_transformer,
                     bonded_parameters,
                     protein_info.loc[name, 'temperature'],
                     jacobian = True)

xml = omm.XmlSerializer.serialize(system)
f = open(f"./output/{name}/system_im.xml", 'w')
f.write(xml)
f.close()

exit()
