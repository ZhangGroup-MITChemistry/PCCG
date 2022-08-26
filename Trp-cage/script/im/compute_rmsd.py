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
traj_im = mdtraj.load_dcd(f"./output/{name}/im/CG_simulations/traj.dcd", psf)
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)

helix_particle_index = set(range(2, 14))
    
rmsd_helix_md = mdtraj.rmsd(traj_md, traj_ref, atom_indices = list(helix_particle_index))
rmsd_helix_im = mdtraj.rmsd(traj_im, traj_ref, atom_indices = list(helix_particle_index))

rmsd_md = mdtraj.rmsd(traj_md, traj_ref)
rmsd_im = mdtraj.rmsd(traj_im, traj_ref)

fig = plt.figure(0)
fig.clf()
plt.hist(rmsd_helix_md, bins = 40, alpha = 0.5, label = 'md', density = True)
plt.hist(rmsd_helix_im, bins = 40, alpha = 0.5, label = 'im', density = True)
plt.savefig(f"./output/{name}/im/rmsd_helix_hist.pdf")

fig = plt.figure(0)
fig.clf()
plt.hist(rmsd_md, bins = 40, alpha = 0.5, label = 'md', density = True)
plt.hist(rmsd_im, bins = 40, alpha = 0.5, label = 'im', density = True)
plt.savefig(f"./output/{name}/im/rmsd_hist.pdf")

