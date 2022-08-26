import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from functools import reduce
import argparse
import pickle
import mdtraj
import torch
torch.set_default_dtype(torch.double)
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
from CLCG.utils.splines import *
from CLCG.utils.CL import *
from sys import exit
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, default = '2JOF')
args = parser.parse_args()

name = args.name

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
traj_im = mdtraj.load_dcd(f"./output/{name}/im/CG_simulations/traj.dcd", psf)

helix_particle_index = set(range(2, 14))
    
rmsd_helix_md = mdtraj.rmsd(traj_md, traj_ref, atom_indices = list(helix_particle_index))
rmsd_helix_im = mdtraj.rmsd(traj_im, traj_ref, atom_indices = list(helix_particle_index))

rmsd_max = max(rmsd_helix_md.max(), rmsd_helix_im.max())
boundary_knots = [0, rmsd_max]
internal_knots = np.linspace(0, rmsd_max, num = 15, endpoint = False)[1:]

rmsd_helix_noise = np.random.uniform(0, rmsd_max, rmsd_helix_md.shape[0])

basis_md = bs(rmsd_helix_md, internal_knots, boundary_knots)
basis_im = bs(rmsd_helix_im, internal_knots, boundary_knots)
basis_noise = bs(rmsd_helix_noise, internal_knots, boundary_knots)

log_q_noise = np.zeros_like(rmsd_helix_noise)
log_q_md = np.zeros_like(rmsd_helix_md)
log_q_im = np.zeros_like(rmsd_helix_im)

alphas_md, F = contrastive_learning_numpy(log_q_noise,log_q_md,
                                          basis_noise,basis_md)

alphas_im, F = contrastive_learning_numpy(log_q_noise,log_q_im,
                                          basis_noise,basis_im)

rmsd_over_the_range = np.linspace(0.0, rmsd_max, 1000)    
basis_over_the_range = bs(rmsd_over_the_range, internal_knots, boundary_knots)
U_md = np.matmul(basis_over_the_range, alphas_md)
U_im = np.matmul(basis_over_the_range, alphas_im)

with open(f"./output/{name}/im_us/pmf_rmsd_helix.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
pmf_rmsd_helix = data['pmf']
rmsd_helix = data['rmsd']

spline = CubicSpline(rmsd_helix, pmf_rmsd_helix, bc_type = 'natural')
fitted_pmf = spline(rmsd_over_the_range)

U_im = U_im - U_im.min() + fitted_pmf.min()

fig = plt.figure(0)
fig.clf()
plt.plot(rmsd_helix, pmf_rmsd_helix, '-o', label = 'empirical')
plt.plot(rmsd_over_the_range[::50], fitted_pmf[::50], '-o', label = 'fit')
plt.plot(rmsd_over_the_range[::50], U_im[::50], '-o', label = 'icl')
plt.legend()
plt.savefig(f"./output/{name}/rmsd/fitted_pmf_rmsd_helix.pdf")

U = U_md - fitted_pmf
with open(f"./output/{name}/rmsd/rmsd_U.pkl", 'wb') as file_handle:
    pickle.dump({'rmsd_over_the_range': rmsd_over_the_range,
                 'rmsd_min': 0, 'rmsd_max': rmsd_max,
                 'U': U}, file_handle)

exit()

os.makedirs(f"./output/{name}/rmsd", exist_ok = True)
pdf = PdfPages(f"./output/{name}/rmsd/rmsd_helix_cl.pdf")

fig = plt.figure(0)
fig.clf()
axes = plt.hist(rmsd_helix_md, bins = 50, range = (0.0, rmsd_max),
                density = True,
                log = False, label = 'md', alpha = 0.5)
p = np.exp(-(U_md - F))
p = p*axes[0].max()/p.max()
plt.plot(rmsd_over_the_range, p, '-', label = 'cl_md', linewidth = 2)

axes = plt.hist(rmsd_helix_im, bins = 50, range = (0.0, rmsd_max),
                density = True,
                log = False, label = 'im', alpha = 0.5)
p = np.exp(-(U_im - F))
p = p*axes[0].max()/p.max()
plt.plot(rmsd_over_the_range, p, '-', label = 'cl_im', linewidth = 2)

plt.legend()
pdf.savefig()

pdf.close()

    
