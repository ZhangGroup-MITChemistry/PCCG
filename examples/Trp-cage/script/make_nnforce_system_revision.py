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
import os
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
import itertools
import torch
torch.set_default_dtype(torch.double)
sys.path.append("./script")
from NNForce import *
from openmmtorch import TorchForce


name = '2JOF'
weight_decay_list = [1e-3]
# full_include_rmsd = True
# full_weight_decay = 2e-7

full_include_rmsd = False
full_weight_decay = 4e-7

for weight_decay in weight_decay_list:

    ## read system
    with open(f"./output/{name}/full_system/rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay:.3E}.xml", 'r') as file_handle:
        xml = file_handle.read()    
    system = omm.XmlSerializer.deserialize(xml)

    protein_info = pd.read_csv(
        "./info/protein_temperature_and_ionic_strength.txt",
        index_col = 'name',
        comment = '#')
    T = protein_info.loc[name, 'temperature']

    with open(f"./output/{name}/coor_transformer.pkl", 'rb') as file_handle:
        coor_transformer = pickle.load(file_handle)

    data = torch.load(f"./output/{name}/FF/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_NNForce_weight_decay_{weight_decay:.3E}_different_initial_parameters.pth")
    hidden_size = data['hidden_size']
    state_dict = data['state_dict']
    p_index_for_distance = data['p_index_for_distances']
    
    p_index_for_angle = []
    p_index_for_angle.append([coor_transformer.ref_particle_2,
                              coor_transformer.ref_particle_1,
                              coor_transformer.ref_particle_3])
    for p in coor_transformer.particle_visited_in_order:
        p_index_for_angle.append(coor_transformer.angle_particle_idx[p])
    p_index_for_angle = torch.LongTensor(p_index_for_angle)

    p_index_for_dihedral = []
    for p in coor_transformer.particle_visited_in_order:
        p_index_for_dihedral.append(coor_transformer.dihedral_particle_idx[p])
    p_index_for_dihedral = torch.LongTensor(p_index_for_dihedral)

    num_particles = coor_transformer.num_particles
    nnforce = NNForce(num_particles,
                      hidden_size,
                      p_index_for_angle,
                      p_index_for_dihedral,
                      p_index_for_distance,
                      T)
    nnforce.nnforce_net.load_state_dict(state_dict)

    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)
    xyz = torch.from_numpy(traj_ref.xyz[0]).double()

    traced_nnforce = torch.jit.trace(nnforce, xyz)
    traced_nnforce.save(f"./output/{name}/FF/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_NNForceXYZ_weight_decay_{weight_decay:.3E}_different_initial_parameters.pth")

    nnforce = TorchForce(f"./output/{name}/FF/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_NNForceXYZ_weight_decay_{weight_decay:.3E}_different_initial_parameters.pth")
    nnforce.setForceGroup(0)
    system.addForce(nnforce)

    xml = omm.XmlSerializer.serialize(system)
    os.makedirs(f"./output/{name}/nnforce_system", exist_ok = True)
    with open(f"./output/{name}/nnforce_system/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_NNForce_weight_decay_{weight_decay:.3E}_different_initial_parameters.xml", 'w') as file_handle:
        file_handle.write(xml)

    fricCoef = 1./unit.picoseconds
    stepsize = 5. * unit.femtoseconds
    integrator = omm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

    platform = omm.Platform.getPlatformByName('Reference')
    context = omm.Context(system, integrator, platform)
    context.setPositions(xyz.numpy())
    state = context.getState(getEnergy = True)
    potential_energy = state.getPotentialEnergy()

    print(f"weight_decay: {weight_decay:.3E}", data['loss_wo_penalty'], potential_energy, flush = True)
exit()
