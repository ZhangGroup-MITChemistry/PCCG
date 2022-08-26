#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=make_system
#SBATCH --time=01:00:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-11
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

# weight_decay_list = [1e-8,
#                      1e-7, 2e-7, 5e-7,
#                      1e-6, 2e-6, 5e-6,
#                      1e-5]

weight_decay_list = [5e-7,
                     1e-6, 2e-6]


job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
name = protein_names.iloc[job_idx, 0]

print(f"name: {name}", flush = True)

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

AA_info = pd.read_csv("./info/amino_acids_with_learned_sigmas.csv",
                      index_col = 'name')
masses = [AA_info.loc[r.name, 'mass'] for r in psf.residues]


def make_custom_elec_force(elec_type):
    if elec_type == 'simple':
        formula = ["ONE_4PI_EPS0*charge1*charge2/(r*epsilon)",
                   "ONE_4PI_EPS0 = 138.935456",
                   "epsilon = 78.4"
        ]
    elif elec_type == 'fshift':
        formula = ["ONE_4PI_EPS0*charge1*charge2/epsilon*fr*step(rc - r)",
                   "fr = 1/r - 5/(3*rc) + 5*r^3/(3*rc^4) - r^4/rc^5",              
                   "ONE_4PI_EPS0 = 138.935456",
                   "epsilon = 15",
                   "rc = 1.2"
        ]
    elif elec_type == 'DH':
        formula = ["ONE_4PI_EPS0*charge1*charge2/(r*eps)*exp(-r/lambda_D)*step(r-r_cut)",
                   "lambda_D = sqrt(eps*constant*T/ionic_strength)",
                   "constant = 2.3811197070710097 / 10^6",
                   "eps = A + B/(1 + kappa*exp(-lambda*B*r))",
                   "B = 78.4 - A; A = -8.5525; kappa = 7.7839; lambda = 0.03627",
                   "ONE_4PI_EPS0 = 138.935456",
                   "r_cut = 0.326"
        ]
    elif elec_type == 'DH_2':
        formula = ["ONE_4PI_EPS0*charge1*charge2/(r*eps)*exp(-r/lambda_D)",
                   "lambda_D = sqrt(eps*constant*T/ionic_strength)",
                   "constant = 2.3811197070710097 / 10^6",
                   "eps = 78.4",
                   "ONE_4PI_EPS0 = 138.935456",
        ]
    else:
        raise ValueError("elec_type has to be either simple, fshift, DH, or DH_2")
    
    custom_elec_force = omm.CustomNonbondedForce(";".join(formula))
    custom_elec_force.addPerParticleParameter('charge')

    if elec_type == 'DH' or 'DH_2':    
        custom_elec_force.addGlobalParameter("T", T)
        custom_elec_force.addGlobalParameter("ionic_strength", ionic_strength)
    
    return custom_elec_force

def sort_pair_name(pair):
    if pair[0] <= pair[1]:
        return pair
    else:
        tmp = pair[0]
        pair[0] = pair[1]
        pair[1] = tmp
        return pair

for weight_decay in weight_decay_list:
    #for elec_type, ss_type in itertools.product(['simple', 'fshift'], ['simple', 'extended']):
    #for elec_type, ss_type in itertools.product(['DH'], ['simple', 'extended']):
    for elec_type, ss_type in itertools.product(['DH_2'], ['simple']):                
        print(f"weight_decay: {weight_decay:.3E}, elec_type: {elec_type:8}, ss_type: {ss_type:8}")
        data = torch.load(f"./output/{name}/FF/FF_elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.pt",
                          map_location=torch.device('cpu'))
        FF = data['FF']
        bonded_ff, lj_ff, rmsd_ff = FF['bonded'], FF['lj'], FF['rmsd']

        ## bonded terms
        system = make_system(masses,
                             coor_transformer,
                             bonded_ff,
                             protein_info.loc[name, 'temperature'],
                             jacobian = False)

        ## electrostatic
        custom_elec_force = make_custom_elec_force(elec_type)
        custom_elec_force.setForceGroup(1)
        for i in range(psf.n_residues):
            resname = psf.residue(i).name
            charge = AA_info.loc[resname, 'charge']
            custom_elec_force.addParticle([charge])
        bonds = [[i, i+1] for i in range(psf.n_residues - 1)]
        custom_elec_force.createExclusionsFromBonds(bonds, 3)    
        system.addForce(custom_elec_force)        

        ## LJ
        with open(f"./output/common/LJ_rmin.pkl", 'rb') as file_handle:
            r_min_dict = pickle.load(file_handle)
        aa_pairs = list(r_min_dict.keys())
        aa_pairs.sort()

        ulj = lj_ff['U'].numpy()
        lj_min = lj_ff['min']
        lj_max = lj_ff['max']    
        func = omm.Continuous2DFunction(xsize = ulj.shape[0],
                                        ysize = ulj.shape[1],
                                        values = ulj.reshape(-1, order = 'F'),
                                        xmin = 0., xmax = float(ulj.shape[0] - 1),
                                        ymin = lj_min, ymax = lj_max)

        LJ_force = omm.CustomCompoundBondForce(2, f"U_LJ(idx, r); r = distance(p1,p2)")
        LJ_force.addTabulatedFunction(f"U_LJ", func)
        LJ_force.addPerBondParameter('idx')

        for i in range(len(resnames)):
            for j in range(i+4, len(resnames)):
                aa1, aa2 = resnames[i], resnames[j]
                pair = tuple(sort_pair_name([aa1, aa2]))
                idx = float(aa_pairs.index(pair))
                LJ_force.addBond([i,j], [idx])
        LJ_force.setForceGroup(2)        
        system.addForce(LJ_force)

        ## rmsd
        ss = pd.read_table(f"./info/secondary_structure_{ss_type}.csv",
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
        with open(f"./output/{name}/full_system/elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.xml", 'w') as file_handle:
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
