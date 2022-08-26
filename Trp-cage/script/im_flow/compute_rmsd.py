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
traj_flow = mdtraj.load_dcd(f"./output/{name}/im_flow/traj.dcd", psf)
traj_sim = mdtraj.load_dcd(f"./output/{name}/im_flow/CG_simulations/traj.dcd", psf)
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)

with open(f"./output/{name}/flow/helix_index.pkl", 'rb') as file_handle:
    helix_index = pickle.load(file_handle)

rmsd_helix_md = mdtraj.rmsd(traj_md, traj_ref, atom_indices = list(helix_index['helix_particle_index']))
rmsd_helix_flow = mdtraj.rmsd(traj_flow, traj_ref, atom_indices = list(helix_index['helix_particle_index']))
rmsd_helix_sim = mdtraj.rmsd(traj_sim, traj_ref, atom_indices = list(helix_index['helix_particle_index']))

rmsd_md = mdtraj.rmsd(traj_md, traj_ref)
rmsd_flow = mdtraj.rmsd(traj_flow, traj_ref)
rmsd_sim = mdtraj.rmsd(traj_sim, traj_ref)

fig = plt.figure(0)
fig.clf()
plt.hist(rmsd_helix_md, bins = 40, alpha = 0.5, label = 'md', density = True)
plt.hist(rmsd_helix_flow, bins = 40, alpha = 0.5, label = 'flow', density = True)
plt.hist(rmsd_helix_sim, bins = 40, alpha = 0.5, label = 'sim', density = True)
plt.savefig(f"./output/{name}/im_flow/rmsd_helix_hist.pdf")

fig = plt.figure(0)
fig.clf()
plt.hist(rmsd_md, bins = 40, alpha = 0.5, label = 'md', density = True)
plt.hist(rmsd_flow, bins = 40, alpha = 0.5, label = 'flow', density = True)
plt.hist(rmsd_sim, bins = 40, alpha = 0.5, label = 'sim', density = True)
plt.savefig(f"./output/{name}/im_flow/rmsd_hist.pdf")

