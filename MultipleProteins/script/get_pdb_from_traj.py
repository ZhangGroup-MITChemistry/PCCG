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
import os

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
weight_decay_list = [1e-10, 1e-9, 1e-8,
                     1e-7, 1e-6, 1e-5,
                     1e-4, 1e-3]
protein_names = protein_names.iloc[:, 0].tolist()

name = 'A3D'
weight_decay = 1e-5

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")    
traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)    
traj_md = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", psf, stride = 10)
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/weight_decay_{weight_decay:.3E}.dcd", psf, stride = 50)

os.makedirs(f"./output/test", exist_ok = True)
traj_ref.save_pdb(f"./output/test/ref.pdb")
os.makedirs(f"./output/test/md/", exist_ok = True)

# width = len(str(len(traj_md)))
# for i in range(len(traj_md)):
#     folder_name = i // 1000
#     os.makedirs(f"./output/test/md/{folder_name:04}", exist_ok = True)
#     traj_md[i].save_pdb(f"./output/test/md/{folder_name:04}/md_{i:0{width}}.pdb")
#     if (i + 1) % 100 == 0:
#         print(i)

width = len(str(len(traj_cg)))
for i in range(len(traj_cg)):
    folder_name = i // 1000
    os.makedirs(f"./output/test/cg/{folder_name:04}", exist_ok = True)
    traj_cg[i].save_pdb(f"./output/test/cg/{folder_name:04}/md_{i:0{width}}.pdb")
    if (i + 1) % 100 == 0:
        print(i)
        
exit()

@ray.remote    
def get_rmsd_cg(name, weight_decay):
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    traj_ref = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)
    traj_cg_nvt = mdtraj.load_dcd(f"./output/{name}/NVT/weight_decay_{weight_decay:.3E}.dcd", psf, stride = 10)
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
