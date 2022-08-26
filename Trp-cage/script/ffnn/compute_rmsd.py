import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
args = parser.parse_args()

name = args.name
print(f"name: {name}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
traj_im = mdtraj.load_dcd(f"./output/{name}/im/CG_simulations/traj.dcd", psf)

weight_decay_list = [1e-9, 5e-9,
                     1e-8, 5e-8,
                     1e-7, 5e-7,
                     1e-6, 5e-6,
                     1e-5, 5e-5,
                     1e-4, 5e-4,
                     1e-3, 5e-3]

traj_ffnn = {}
for weight_decay in weight_decay_list:
    traj = mdtraj.load_dcd(f"./output/{name}/ffnn/CG_simulations/traj_weight_decay_{weight_decay:.3E}.dcd", psf)
    traj_ffnn[weight_decay] = traj
    
helix_particle_index = set(range(2, 14))
    
rmsd_helix_md = mdtraj.rmsd(traj_md, traj_ref, atom_indices = list(helix_particle_index))
rmsd_helix_im = mdtraj.rmsd(traj_im, traj_ref, atom_indices = list(helix_particle_index))
rmsd_helix_ffnn = {}
for weight_decay in weight_decay_list:
    rmsd_helix_ffnn[weight_decay] = mdtraj.rmsd(traj_ffnn[weight_decay], traj_ref, atom_indices = list(helix_particle_index))

rmsd_md = mdtraj.rmsd(traj_md, traj_ref)
rmsd_im = mdtraj.rmsd(traj_im, traj_ref)
rmsd_ffnn = {}
for weight_decay in weight_decay_list:
    rmsd_ffnn[weight_decay] = mdtraj.rmsd(traj_ffnn[weight_decay], traj_ref)

with PdfPages(f"./output/{name}/ffnn/rmsd_helix_hist.pdf") as pdf:
    for weight_decay in weight_decay_list:
        fig = plt.figure()
        fig.clf()
        plt.hist(rmsd_helix_md, bins = 40, alpha = 0.5, label = 'md', density = True)
        plt.hist(rmsd_helix_im, bins = 40, alpha = 0.5, label = 'im', density = True)
        plt.hist(rmsd_helix_ffnn[weight_decay], bins = 40, alpha = 0.5, label = 'ffnn', density = True)
        plt.title(f"weight_decay: {weight_decay:.3E}")
        plt.legend()
        pdf.savefig()
        plt.close()

with PdfPages(f"./output/{name}/ffnn/rmsd_hist.pdf") as pdf:
    for weight_decay in weight_decay_list:
        fig = plt.figure()
        fig.clf()
        plt.hist(rmsd_md, bins = 40, alpha = 0.5, label = 'md', density = True)
        plt.hist(rmsd_im, bins = 40, alpha = 0.5, label = 'im', density = True)
        plt.hist(rmsd_ffnn[weight_decay], bins = 40, alpha = 0.5, label = 'ffnn', density = True)
        plt.title(f"weight_decay: {weight_decay:.3E}")
        plt.legend()        
        pdf.savefig()
        plt.close()
        
