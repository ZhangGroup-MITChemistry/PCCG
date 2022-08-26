#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=make_system
#SBATCH --time=01:00:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/make_system_%a.txt

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
import itertools

protein_names = pd.read_csv("./info/protein_names.txt",
                            comment = "#",
                            header = None)
protein_names = protein_names.iloc[0:2, 0].tolist()
flag_rmsd = [False, True]
# weight_decay_list = [1e-7, 5e-7,
#                      1e-6, 5e-6,
#                      1e-5, 5e-5,
#                      1e-4, 5e-4,
#                      1e-3, 5e-3]
weight_decay_list = [2e-7, 3e-7, 4e-7]
options = list(itertools.product(protein_names, flag_rmsd, weight_decay_list))

for rank in range(len(options)):    
    name, include_rmsd, weight_decay = options[rank]
    print(f"name: {name}, rmsd: {include_rmsd}, weight_decay: {weight_decay}", flush = True)

    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    resnames = [residue.name for residue in psf.residues]
    for i in range(len(resnames)):
        if resnames[i] == "NLE":
            resnames[i] = "ILE"
    nres = len(resnames)

    protein_info = pd.read_csv("./info/protein_temperature_and_ionic_strength.txt",
                               index_col = 'name',
                               comment = '#')
    T = protein_info.loc[name, 'temperature']
    ionic_strength = protein_info.loc[name, 'ionic_strength']
    
    kbT = unit.BOLTZMANN_CONSTANT_kB * T *unit.kelvin * unit.AVOGADRO_CONSTANT_NA
    kbT_kJ_per_mole = kbT.value_in_unit(unit.kilojoule_per_mole)
    beta = 1/kbT_kJ_per_mole

    Kb = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
    Kb = Kb.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)

    with open(f"./output/{name}/coor_transformer.pkl", 'rb') as file_handle:
        coor_transformer = pickle.load(file_handle)

    AA_info = pd.read_csv("./info/amino_acids_with_learned_sigmas.csv", index_col = 'name')
    masses = [AA_info.loc[r.name, 'mass'] for r in psf.residues]

    data = torch.load(f"./output/{name}/FF/FF_rmsd_{include_rmsd}_weight_decay_{weight_decay:.3E}.pt")
    FF = data['FF']
    bonded_ff, lj_ff, rmsd_ff = FF['bonded'], FF['lj'], FF['rmsd']

    ## bonded terms
    system = make_system(masses,
                         coor_transformer,
                         bonded_ff,
                         protein_info.loc[name, 'temperature'],
                         jacobian = False)

    ## electrostatic
    def make_custom_elec_force(T, ionic_strength):
        formula = ["ONE_4PI_EPS0*charge1*charge2/(r*eps)*exp(-r/lambda_D)",
                   "lambda_D = sqrt(eps*constant*T/ionic_strength)",
                   "constant = 2.3811197070710097 / 10^6",
                   "eps = 78.4",
                   "ONE_4PI_EPS0 = 138.935456",
        ]

        custom_elec_force = omm.CustomNonbondedForce(";".join(formula))
        custom_elec_force.addGlobalParameter("T", T)
        custom_elec_force.addGlobalParameter("ionic_strength", ionic_strength)
        custom_elec_force.addPerParticleParameter('charge')
        return custom_elec_force

    custom_elec_force = make_custom_elec_force(T, ionic_strength)
    custom_elec_force.setForceGroup(1)
    for i in range(psf.n_residues):
        resname = psf.residue(i).name
        charge = AA_info.loc[resname, 'charge']
        custom_elec_force.addParticle([charge])
    bonds = [[i, i+1] for i in range(psf.n_residues - 1)]
    custom_elec_force.createExclusionsFromBonds(bonds, 3)
    system.addForce(custom_elec_force)
    
    ## LJ
    aa_pairs = [(i,j) for i in range(psf.n_residues)
                      for j in range(i+4, psf.n_residues)]    
    aa_pairs.sort()

    ulj = lj_ff['U'].numpy()
    lj_min = lj_ff['min']
    lj_max = lj_ff['max']    
    func = omm.Continuous2DFunction(xsize = ulj.shape[0],
                                    ysize = ulj.shape[1],
                                    values = ulj.reshape(-1, order = 'F'),
                                    xmin = 0., xmax = float(ulj.shape[0] - 1),
                                    ymin = lj_min, ymax = lj_max)

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

    ## rmsd
    if include_rmsd:
        ss = pd.read_table("./info/secondary_structure.csv",
                           index_col = 'name', header = 0)
        ss = eval(ss.loc[name, 'ss'])
        traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)

        rmsd_min = rmsd_ff['min']
        rmsd_max = rmsd_ff['max']
        rmsd_U = rmsd_ff['U']

        for idx_ss in range(len(ss)):
            particle_index = [list(range(start-1, end)) for start, end in ss[idx_ss]]
            particle_index = list(itertools.chain(*particle_index))

            rmsd_force = omm.RMSDForce(traj_ref.xyz[0], particle_index)
            cv_force = omm.CustomCVForce("Kb*T*u_rmsd(rmsd)")
            cv_force.addCollectiveVariable('rmsd', rmsd_force)
            u_rmsd = omm.Continuous1DFunction(rmsd_U[idx_ss].numpy(),
                                              rmsd_min[idx_ss].item(),
                                              rmsd_max[idx_ss].item())
            cv_force.addTabulatedFunction('u_rmsd', u_rmsd)
            cv_force.addGlobalParameter('Kb', Kb)
            cv_force.addGlobalParameter('T', T)
            system.addForce(cv_force)

    xml = omm.XmlSerializer.serialize(system)
    os.makedirs(f"./output/{name}/full_system", exist_ok = True)
    with open(f"./output/{name}/full_system/rmsd_{include_rmsd}_weight_decay_{weight_decay:.3E}.xml", 'w') as file_handle:
        file_handle.write(xml)

    fricCoef = 1./unit.picoseconds
    stepsize = 2. * unit.femtoseconds
    integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

    platform = omm.Platform.getPlatformByName('Reference')
    context = omm.Context(system, integrator, platform)
    traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)
    context.setPositions(traj_ref.xyz[0])
    state = context.getState(getEnergy = True)
    potential_energy = state.getPotentialEnergy()
    print(potential_energy)

exit()
