import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
import numpy as np
import torch
import pyemma
import mdtraj
import os
from scipy.stats import gaussian_kde
from sys import exit
from matplotlib import cm
import pickle
from matplotlib.backends.backend_pdf import PdfPages

name = "2JOF"
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
resnames = [residue.name for residue in psf.residues]
for i in range(len(resnames)):
    if resnames[i] == "NLE":
        resnames[i] = "ILE"

## load spline coefficients
data = torch.load(f"./output/{name}/FF/FF_rmsd_True_weight_decay_{2e-7:.3E}.pt")
U = data['FF']['lj']['U'].numpy()
r_min, r_max = data['FF']['lj']['min'], data['FF']['lj']['max']
r = np.linspace(r_min, r_max, U.shape[1])
pdf = PdfPages(f"./output/plots/{name}_lj_all.pdf")
for i in range(U.shape[0]):
    fig = plt.figure()
    plt.plot(r, U[i])
    plt.ylim(-3, 5)
    plt.title(f"index_{i}")
    pdf.savefig()
    print(i)
pdf.close()    

pdf = PdfPages(f"./output/plots/{name}_lj_selected.pdf")
fig = plt.figure()
for i in [103, 123, 128]:    
    plt.plot(r, U[i], linewidth = 4, alpha = 0.6)
plt.ylim(-3, 5)
pdf.savefig()

fig = plt.figure()
plt.plot(r, U[108], linewidth = 4)
plt.ylim(-3, 5)
pdf.savefig()

pdf.close()
exit()


nrow = 6
ncol = 4

fig = plt.figure(np.random.randint(0, 1000), figsize = (6.4*ncol, 4.8*nrow))
fig.clf()
for k in range(len(pair_indices)):
    plt.subplot(nrow, ncol, k+1)
    pair = pair_indices[k]
    i,j = pair_indices[k]

    aa1, aa2 = resnames[i], resnames[j]
    if resnames[j] < resnames[i]:
        aa1, aa2 = resnames[j], resnames[i]
    
    r_min, r_max = LJ_parameters[pair]['r_min'], LJ_parameters[pair]['r_max']
    ulj = LJ_parameters[pair]['ulj']
    r = np.linspace(r_min , r_max, len(ulj))
    plt.plot(r, ulj, linewidth = 6.0)
    plt.xlim(0, r_max)
    plt.ylim(ulj.min()-1, 7)

    plt.title(f"{i}-{j}:{aa1}-{aa2}")
    print(k)

plt.tight_layout()    
fig.savefig(f"./output/plots/{name}_lj.eps")

exit()
