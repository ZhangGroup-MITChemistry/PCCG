import torch
import numpy as np
import pickle
import os
import argparse
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import transform, MMFlow, utils
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
args = parser.parse_args()

name = args.name

with open(f"./output/{name}/md/coor_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)

helix_particle_index = set(range(2,14))
helix_angle_index = []
helix_angle_particle_index = []
helix_dihedral_index = []
helix_dihedral_particle_index = []
for i in range(len(coor_transformer.particle_visited_in_order)):
    p = coor_transformer.particle_visited_in_order[i]
    p1, p2, p3 = coor_transformer.angle_particle_idx[p]
    
    if p1 in helix_particle_index and \
       p2 in helix_particle_index and \
       p3 in helix_particle_index:
        helix_angle_index.append(i)
        helix_angle_particle_index.append([p1, p2, p3])

    p1, p2, p3, p4 = coor_transformer.dihedral_particle_idx[p]    
    if p1 in helix_particle_index and \
       p2 in helix_particle_index and \
       p3 in helix_particle_index and \
       p4 in helix_particle_index:
        helix_dihedral_index.append(i)
        helix_dihedral_particle_index.append([p1, p2, p3, p4])
        
helix_reference_particle_3_angle_flag = False

if coor_transformer.ref_particle_1 in helix_particle_index and \
   coor_transformer.ref_particle_2 in helix_particle_index and \
   coor_transformer.ref_particle_3 in helix_particle_index:
    helix_reference_particle_3_angle_flag = True

os.makedirs(f"./output/{name}/ffnn", exist_ok = True)    
with open(f"./output/{name}/ffnn/helix_index.pkl", 'wb') as file_handle:
    pickle.dump({'helix_particle_index': helix_particle_index,
                 'helix_angle_index': helix_angle_index,
                 'helix_angle_particle_index': helix_angle_particle_index,
                 'helix_dihedral_index': helix_dihedral_index,
                 'helix_dihedral_particle_index': helix_dihedral_particle_index,
                 'helix_reference_particle_3_angle_flag': helix_reference_particle_3_angle_flag},
                file_handle)



