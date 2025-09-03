import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sys import exit
import itertools
import pandas as pd

weight_decay_list = [0.0,
                     1e-10, 5e-10,  
                     1e-9 , 5e-9 ,
                     1e-8 , 5e-8 ,
                     1e-7 , 5e-7 ,
                     1e-6 , 5e-6 ,
                     1e-5 , 5e-5 ,
                     1e-4 , 5e-4 ,
                     1e-3 , 5e-3 ,
                     1e-2 , 5e-2 ,
                     1e-1 ]

## read system
name = "2JOF"
# full_include_rmsd = True
# full_weight_decay = 2e-7

full_include_rmsd = False
full_weight_decay = 4e-7


psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)    
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
rmsd_md = mdtraj.rmsd(traj_md, traj_ref)

pdf_hist =  PdfPages(f"./output/{name}/rmsd_full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay:.3E}_nnforce_hist.pdf")
pdf_traj =  PdfPages(f"./output/{name}/rmsd_full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay:.3E}_nnforce_traj.pdf")

for weight_decay in weight_decay_list:

    print(f"name: {name}, weight_decay: {weight_decay}", flush = True)

    traj_cg_NVT = mdtraj.load_dcd(f"./output/{name}/nnforce_NVT/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_weight_decay_{weight_decay:.3E}.dcd", psf)
    rmsd_cg_NVT = mdtraj.rmsd(traj_cg_NVT, traj_ref)        

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
    plt.title(f"weight_decay: {weight_decay:.3E}")
    pdf_traj.savefig()
    plt.close()

pdf_hist.close()
pdf_traj.close()

exit()
