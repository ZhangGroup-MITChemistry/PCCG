import mdtraj
import argparse
import math
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, default = '2JOF')
parser.add_argument('--size', type = int, default = 48)
args = parser.parse_args()

name = args.name
size = args.size
print(f"name: {name}, size: {size}")

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_ref = mdtraj.load_xyz(f"./output/{name}/md/reference_structure.xyz", psf)

rmsd = []
for rank in range(size):
    traj = mdtraj.load_dcd(f"./output/{name}/im_us/trajs/traj_{rank}.dcd", psf, stride = 1)
    rmsd.append(mdtraj.rmsd(traj, traj_ref))
    print(rank)
    
fig = plt.figure(10, figsize = (6.4*4, 4.8))
fig.clf()
for rank in range(0, size):
    plt.hist(rmsd[rank], 30, density = True, alpha = 0.4)
plt.savefig(f"./output/{name}/im_us/rmsd_hist.pdf")
