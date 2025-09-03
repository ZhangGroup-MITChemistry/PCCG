import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sys import exit
from itertools import product
import pandas as pd
import ray

parser = argparse.ArgumentParser()
parser.add_argument('--elec_type', type = str)
parser.add_argument('--ss_type', type = str)
args = parser.parse_args()

elec_type = args.elec_type
ss_type = args.ss_type

print(f"elec_type: {elec_type:10}, ss_type: {ss_type}")

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
# weight_decay_list = [1e-8,
#                      1e-7, 2e-7, 5e-7,
#                      1e-6, 2e-6, 5e-6,
#                      1e-5]

weight_decay_list = [5e-7,
                     1e-6, 2e-6]

protein_names = protein_names.iloc[:, 0].tolist()

pdf_hist =  PdfPages(f"./output/plots/rmsd_full_hist_elec_type_{elec_type}_ss_type_{ss_type}.pdf")
pdf_traj =  PdfPages(f"./output/plots/rmsd_full_traj_elec_type_{elec_type}_ss_type_{ss_type}.pdf")

rmsd_md_dict = {}
traj_ref_dict = {}
for name in protein_names:
    print(name)
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")    
    traj_ref_dict[name] = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)    
    traj_md = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", psf, stride = 10)
    rmsd_md_dict[name] = mdtraj.rmsd(traj_md, traj_ref_dict[name])

@ray.remote    
def get_rmsd_cg(name, weight_decay):
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)
    traj_cg_nvt = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_{elec_type}_ss_type_{ss_type}_weight_decay_{weight_decay:.3E}.dcd",
                                  psf, stride = 10)
    rmsd_cg_nvt = mdtraj.rmsd(traj_cg_nvt, traj_ref)
    return rmsd_cg_nvt

#ray.shutdown()
ray.init(ignore_reinit_error = True, _temp_dir = "/home/gridsan/dingxq/tmp/ray")

rmsd_cg_nvt = ray.get([get_rmsd_cg.remote(name, weight_decay) for name in protein_names for weight_decay in weight_decay_list])
rmsd_cg_nvt_dict = {}
idx = 0
for name in protein_names:
    rmsd_cg_nvt_dict[name] = {}
    print(name)
    for weight_decay in weight_decay_list:
        rmsd_cg_nvt_dict[name][weight_decay] = rmsd_cg_nvt[idx]
        idx += 1

for weight_decay in weight_decay_list:
    print(weight_decay)
    
    ncols_hist = 4
    nrows_hist = 3
    fig_hist = plt.figure(figsize = (6.4*ncols_hist, 4.8*nrows_hist))
    fig_hist.suptitle(f"weight_decay: {weight_decay:.3E}")

    ncols_traj = 1
    nrows_traj = 12
    fig_traj = plt.figure(figsize = (6.4*4, 4.8*nrows_traj))
    fig_traj.suptitle(f"weight_decay: {weight_decay:.3E}")
    
    for name in protein_names:
        print(name)
        rmsd_md = rmsd_md_dict[name]
        rmsd_cg_nvt = rmsd_cg_nvt_dict[name][weight_decay]
        
        axes = fig_hist.add_subplot(nrows_hist, ncols_hist, protein_names.index(name) + 1)
        axes.hist(rmsd_md, bins = 40, alpha = 0.5, label = 'md', density = True)
        axes.hist(rmsd_cg_nvt, bins = 40, alpha = 0.5, label = 'cg_nvt', density = True)
        axes.legend()
        axes.set_title(f"{name}")

        axes = fig_traj.add_subplot(nrows_traj, ncols_traj, protein_names.index(name) + 1)
        axes.plot(rmsd_md[::len(rmsd_md)//10000], alpha = 0.5, label = 'md')
        axes.plot(rmsd_cg_nvt[::len(rmsd_cg_nvt)//10000], alpha = 0.5, label = 'cg_nvt')
        axes.legend()
        axes.set_title(f"{name}")
        
    pdf_hist.savefig(fig_hist)
    pdf_traj.savefig(fig_traj)    

pdf_hist.close()
pdf_traj.close()

exit()

plt.plot(rmsd_md[::len(rmsd_md)//10000], alpha = 0.5, label = 'md')
plt.plot(rmsd_cg_nvt[::len(rmsd_cg_nvt)//10000], alpha = 0.5, label = 'cg_nvt')
plt.legend()
plt.title(f"{name}, weight_decay: {weight_decay:.3E}")
pdf_traj.savefig()
plt.close()

exit()
