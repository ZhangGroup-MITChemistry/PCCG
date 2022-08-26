import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
args = parser.parse_args()

name = args.name
print(f"name: {name}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)
traj_rmsd = mdtraj.load_dcd(f"./output/{name}/rmsd/CG_simulations/traj.dcd", psf)
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)

helix_particle_index = range(2, 14)

rmsd_helix_md = mdtraj.rmsd(traj_md, traj_ref, atom_indices = helix_particle_index)
rmsd_helix_rmsd = mdtraj.rmsd(traj_rmsd, traj_ref, atom_indices = helix_particle_index)

rmsd_md = mdtraj.rmsd(traj_md, traj_ref)
rmsd_rmsd = mdtraj.rmsd(traj_rmsd, traj_ref)

fig = plt.figure(0)
fig.clf()
plt.hist(rmsd_helix_md, bins = 40, alpha = 0.5, label = 'md', density = True)
plt.hist(rmsd_helix_rmsd, bins = 40, alpha = 0.5, label = 'cg', density = True)
plt.legend()
plt.savefig(f"./output/{name}/rmsd/rmsd_helix_hist.pdf")

fig = plt.figure(0)
fig.clf()
plt.hist(rmsd_md, bins = 40, alpha = 0.5, label = 'md', density = True)
plt.hist(rmsd_rmsd, bins = 40, alpha = 0.5, label = 'cg', density = True)
plt.legend()
plt.savefig(f"./output/{name}/rmsd/rmsd_hist.pdf")

fig = plt.figure(1, figsize = (6.4*4, 4.8))
fig.clf()
plt.plot(rmsd_helix_md[::200], '-o', label = 'md')
plt.plot(rmsd_helix_rmsd[::200], '-o', label = 'cg')
plt.legend()
plt.savefig(f"./output/{name}/rmsd/rmsd_helix_along_traj.pdf")
