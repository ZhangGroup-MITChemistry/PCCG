import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sys import exit
import itertools
import pandas as pd

protein_names = pd.read_csv("./info/protein_names.txt",
                            comment = "#",
                            header = None)
protein_names = protein_names.iloc[0:2, 0].tolist()
flag_rmsd = [False, True]
weight_decay_list = [1e-7, 2e-7,
                     3e-7, 4e-7,
                     5e-7,
                     1e-6, 5e-6,
                     1e-5, 5e-5,
                     1e-4, 5e-4,
                     1e-3, 5e-3]
options = list(itertools.product(protein_names, flag_rmsd, weight_decay_list))

for name in protein_names:
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)    
    traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
    rmsd_md = mdtraj.rmsd(traj_md, traj_ref)

    for include_rmsd in flag_rmsd:
        pdf_hist =  PdfPages(f"./output/{name}/rmsd_full_rmsd_{include_rmsd}_hist.pdf")
        pdf_traj =  PdfPages(f"./output/{name}/rmsd_full_rmsd_{include_rmsd}_traj.pdf")

        for weight_decay in weight_decay_list:
            print(f"name: {name}, rmsd: {include_rmsd}, weight_decay: {weight_decay}", flush = True)
    
            traj_cg_NVT = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_{include_rmsd}_weight_decay_{weight_decay:.3E}.dcd", psf)
            rmsd_cg_NVT = mdtraj.rmsd(traj_cg_NVT, traj_ref)        

            # traj_cg_TRE = mdtraj.load_dcd(f"./output/{name}/TRE/weight_decay_{weight_decay:.3E}/traj.dcd", psf)
            # rmsd_cg_TRE = mdtraj.rmsd(traj_cg_TRE, traj_ref)        

            fig = plt.figure()
            fig.clf()

            plt.hist(rmsd_md, bins = 40, alpha = 0.5, label = 'md', density = True)
            plt.hist(rmsd_cg_NVT, bins = 40, alpha = 0.5, label = 'cg_NVT', density = True)

            plt.legend()
            plt.title(f"weight_decay: {weight_decay:.3E}")
            pdf_hist.savefig()
            plt.close()

            fig = plt.figure(figsize = (6.4*4, 4.8))
            fig.clf()

            plt.plot(rmsd_md[::len(rmsd_md)//10000], alpha = 0.5, label = 'md')
            plt.plot(rmsd_cg_NVT[::len(rmsd_cg_NVT)//10000], alpha = 0.5, label = 'cg_NVT')

            plt.legend()
            plt.title(f"rmsd: {include_rmsd}, weight_decay: {weight_decay:.3E}")
            pdf_traj.savefig()
            plt.close()

        pdf_hist.close()
        pdf_traj.close()

exit()
