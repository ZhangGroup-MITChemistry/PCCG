#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=make_system
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-8
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/make_system_%A_%a.txt

import simtk.openmm.app  as app
import simtk.openmm as omm
import simtk.unit as unit
import pickle
import math
from sys import exit
from collections import defaultdict
import numpy as np
import mdtraj
from collections import defaultdict
import pandas as pd
import argparse
from scipy.interpolate import BSpline
import os
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG/")
from CLCG.utils.make_system import *

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
args = parser.parse_args()
name = args.name

weight_decay_list = [1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6]

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
resnames = [residue.name for residue in psf.residues]
for i in range(len(resnames)):
    if resnames[i] == "NLE":
        resnames[i] = "ILE"
nres = len(resnames)

protein_info = pd.read_csv("./script/md/protein_temperature_and_ionic_strength.txt", index_col = 'name', comment = '#')    
T = protein_info.loc[name, 'temperature']
kbT = unit.BOLTZMANN_CONSTANT_kB * T *unit.kelvin * unit.AVOGADRO_CONSTANT_NA
kbT_kJ_per_mole = kbT.value_in_unit(unit.kilojoule_per_mole)
beta = 1/kbT_kJ_per_mole

Kb = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
Kb = Kb.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

AA_info = pd.read_csv("./script/md/amino_acids_with_learned_sigmas.csv", index_col = 'name')
masses = [AA_info.loc[r.name, 'mass'] for r in psf.residues]
    
for weight_decay in weight_decay_list:
    with open(f"./output/{name}/LJ/FF/weight_decay_{weight_decay:.3E}.pkl", 'rb') as file_handle:
        data = pickle.load(file_handle)
        bonded_parameters = data['bonded_parameters']
        LJ_parameters = data['LJ_parameters']    
        rmsd_parameters = data['rmsd_parameters']

    system = make_system(masses, coor_transformer, bonded_parameters, protein_info.loc[name, 'temperature'], jacobian = False)
    aa_pairs = list(LJ_parameters.keys())
    aa_pairs.sort()

    ulj = np.array([LJ_parameters[key]['U'] for key in aa_pairs])
    r_min = LJ_parameters[aa_pairs[0]]['r_min']
    r_max = LJ_parameters[aa_pairs[0]]['r_max']

    func = omm.Continuous2DFunction(xsize = ulj.shape[0],
                                    ysize = ulj.shape[1],
                                    values = ulj.reshape(-1, order = 'F'),
                                    xmin = 0., xmax = float(ulj.shape[0] - 1),
                                    ymin = r_min, ymax = r_max)


    LJ_force = omm.CustomCompoundBondForce(2, f"Kb*T*U_LJ(idx, r); r = distance(p1,p2)")
    LJ_force.addGlobalParameter('Kb', Kb)
    LJ_force.addGlobalParameter('T', T)
    LJ_force.addTabulatedFunction(f"U_LJ", func)
    LJ_force.addPerBondParameter('idx')

    res_names = [res.name for res in psf.residues]    
    for i in range(len(res_names)):
        for j in range(i+4, len(res_names)):
            idx = float(aa_pairs.index((i,j)))
            LJ_force.addBond([i,j], [idx])
    LJ_force.setForceGroup(2)        
    system.addForce(LJ_force)

    traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)
    helix_particle_index = range(2, 14)

    rmsd_force = omm.RMSDForce(traj_ref.xyz[0], helix_particle_index)
    cv_force = omm.CustomCVForce("Kb*T*u_rmsd(rmsd)")
    cv_force.addCollectiveVariable('rmsd', rmsd_force)
    rmsd_min = rmsd_parameters['rmsd_over_range'].min()
    rmsd_max = rmsd_parameters['rmsd_over_range'].max()
    u_rmsd = omm.Continuous1DFunction(rmsd_parameters['U'], rmsd_min, rmsd_max)
    cv_force.addTabulatedFunction('u_rmsd', u_rmsd)
    cv_force.addGlobalParameter('Kb', Kb)
    cv_force.addGlobalParameter('T', T)
    system.addForce(cv_force)

    xml = omm.XmlSerializer.serialize(system)
    os.makedirs(f"./output/{name}/LJ/system", exist_ok = True)
    with open(f"./output/{name}/LJ/system/weight_decay_{weight_decay:.3E}.xml", 'w') as file_handle:
        file_handle.write(xml)

    fricCoef = 1./unit.picoseconds
    stepsize = 2. * unit.femtoseconds
    integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

    platform = omm.Platform.getPlatformByName('Reference')
    context = omm.Context(system, integrator, platform)

    context.setPositions(traj_ref.xyz[0])
    state = context.getState(getEnergy = True)
    potential_energy = state.getPotentialEnergy()
    print(potential_energy)

exit()
