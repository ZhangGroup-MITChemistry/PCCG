import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
args = parser.parse_args()

name = args.name
print(f"name: {name}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
traj_im = mdtraj.load_dcd(f"./output/{name}/im/CG_simulations/traj.dcd", psf)
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)

rmsd_md = mdtraj.rmsd(traj_md, traj_ref)
rmsd_im = mdtraj.rmsd(traj_im, traj_ref)

helix_particle_index = set(range(2, 14))

#weight_decay_list = [1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6]
weight_decay_list = [2e-7]
pdf_hist =  PdfPages(f"./output/{name}/LJ/rmsd_hist.pdf")
pdf_traj =  PdfPages(f"./output/{name}/LJ/rmsd_traj.pdf")
for weight_decay in weight_decay_list:
    traj_lj = mdtraj.load_dcd(f"./output/{name}/LJ/CG_simulations/weight_decay_{weight_decay:.3E}.dcd", psf)
    rmsd_lj = mdtraj.rmsd(traj_lj, traj_ref)

    fig = plt.figure()
    fig.clf()

    plt.hist(rmsd_md, bins = 40, alpha = 0.5, label = 'md', density = True)
    plt.hist(rmsd_lj, bins = 40, alpha = 0.5, label = 'lj', density = True)        

    plt.legend()
    plt.title(f"weight_decay: {weight_decay:.3E}")
    pdf_hist.savefig()
    plt.close()

    fig = plt.figure(figsize = (6.4*4, 4.8))
    fig.clf()

    plt.plot(rmsd_md[::100], alpha = 0.5, label = 'md')
    plt.plot(rmsd_lj[::100], alpha = 0.5, label = 'lj')        

    plt.legend()
    plt.title(f"weight_decay: {weight_decay:.3E}")
    pdf_traj.savefig()
    plt.close()
    
    print(weight_decay)

pdf_hist.close()
pdf_traj.close()
    
exit()
