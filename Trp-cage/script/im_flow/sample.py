import torch
import numpy as np
import pickle
import argparse
import math
import mdtraj

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
args = parser.parse_args()

name = args.name
print(f"name: {name}")

with open(f"./output/{name}/flow/helix_index.pkl", 'rb') as file_handle:
    helix_index = pickle.load(file_handle)

data = torch.load(f"./output/{name}/md/ic.pt")
ic = data['ic']

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

feature_flow = torch.load(f'./output/{name}/flow/feature_from_flow.pt')

if helix_index['helix_reference_particle_3_angle_flag']:
    helix_reference_particle_3_angle = feature_flow[:, 0]*math.pi
    angle = feature_flow[:, 1:1+len(helix_index['helix_angle_index'])]*math.pi
    dihedral = feature_flow[:, 1+len(helix_index['helix_angle_index']):]

ic.reference_particle_3_angle = helix_reference_particle_3_angle
ic.angle[:, helix_index['helix_angle_index']] = angle
ic.dihedral[:, helix_index['helix_dihedral_index']] = dihedral

for j in range(ic.angle.shape[-1]):
    if j not in helix_index['helix_angle_index']:
        ic.angle[:,j] = ic.angle[:,j][torch.randperm(len(ic))]

for j in range(ic.dihedral.shape[-1]):
    if j not in helix_index['helix_dihedral_index']:
        ic.dihedral[:,j] = ic.dihedral[:,j][torch.randperm(len(ic))]
        
xyz, _ = coor_transformer.compute_xyz_from_internal_coordinate(
    ic.reference_particle_1_xyz,
    ic.reference_particle_2_bond,
    ic.reference_particle_3_bond,
    ic.reference_particle_3_angle,
    ic.bond,
    ic.angle,
    ic.dihedral
)
torch.save(ic, f"./output/{name}/im_flow/ic.pt")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_flow = mdtraj.Trajectory(xyz.numpy(), psf)
traj_flow.save_dcd(f"./output/{name}/im_flow/traj.dcd")


    
    
    
    
    
