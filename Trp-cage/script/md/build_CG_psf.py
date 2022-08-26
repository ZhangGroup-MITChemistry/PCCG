#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=build_psf
#SBATCH --time=00:10:00
#SBATCH --partition=xeon-p8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-11
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/build_psf_%a.txt

import time
import argparse
from sys import exit
import mdtraj
import os
import pandas as pd

protein_names = pd.read_csv("./script/md/protein_names.txt", comment = "#", header = None)

job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
name = protein_names.iloc[job_idx, 0]

with open(f"./data/structures/{name}-0_CA_index.txt", 'r') as file_handle:
    line = file_handle.readline()
    line = line.strip()
    fields = line.split()

num_beads = len(fields)
print(f"num_bead: {num_beads}")

pdb = mdtraj.load_pdb(f"./data/structures/{name}-0.pdb")
resnames = [res.name for res in pdb.top.residues]
for i in range(len(resnames)):
    if resnames[i] == "NLE":
        resnames[i] = "ILE"

CA_indices = []        
for atom in pdb.top.atoms_by_name("CA"):
    CA_indices.append(atom.index)

os.makedirs(f"./data/structures/{name}", exist_ok = True)
pdb.atom_slice(CA_indices).save_xyz(f"./data/structures/{name}/{name}.xyz")
file_handle = open(f"./data/structures/{name}/{name}.psf", 'w')

AA_masses = {'ALA':  71.08,
             'ARG': 156.2 ,
             'ASN': 114.1 ,
             'ASP': 115.1 ,
             'CYS': 103.1 ,
             'GLN': 128.1 ,
             'GLU': 129.1 ,
             'GLY':  57.05,
             'HIS': 137.1 ,
             'ILE': 113.2 ,
             'LEU': 113.2 ,
             'LYS': 128.2 ,
             'MET': 131.2 ,
             'PHE': 147.2 ,
             'PRO':  97.12,
             'SER':  87.08,
             'THR': 101.1 ,
             'TRP': 186.2 ,
             'TYR': 163.2 ,
             'VAL':  99.07}


## start line
print("PSF CMAP CHEQ XPLOR", file = file_handle)
print("", file = file_handle)

## title
print("{:>8d} !NTITLE".format(2), file = file_handle)
print("* Coarse Grained System PSF FILE", file = file_handle)
print("* DATE: {} CREATED BY USER: DINGXQ".format(time.asctime()), file = file_handle)

## atoms
print("", file = file_handle)
print("{:>8d} !NATOM".format(num_beads), file = file_handle)
for i in range(1, num_beads+1):
    print("{:>8d} {:<4s} {:<4d} {:<4s} {:<4s} {:<4s} {:>14.6f}{:>14.6f}{:>8d}{:>14.6f}".format(i, "CGM", i, resnames[i-1], "CA", "CA", 0.0, 0.0, 0, 0.0), file = file_handle)

## bonds
print("", file = file_handle)
print("{:>8d} !NBOND: bonds".format(num_beads-1), file = file_handle)
i = 1
count = 0
while i < num_beads:
    print("{:>8d}{:>8d}".format(i,i+1), end = "", file = file_handle)
    count += 1
    if count == 4:
        print("", file = file_handle)
        count = 0
    i += 1
print("", file = file_handle)

## angles
print("", file = file_handle)
print("{:>8d} !NTHETA: angles".format(0), file = file_handle)
print("", file = file_handle)

## diherals
print("", file = file_handle)
print("{:>8d} !NPHI: dihedrals".format(0), file = file_handle)
print("", file = file_handle)

## impropers
print("", file = file_handle)
print("{:>8d} !NIMPHI: impropers".format(0), file = file_handle)
print("", file = file_handle)
file_handle.close()
