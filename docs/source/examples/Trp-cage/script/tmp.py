import os
import pickle
import pandas as pd
import numpy as np

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
protein_names = protein_names.iloc[0:4, 0].tolist()
rmsd_centers_and_k = {}
for name in protein_names:
    with open(f"./output/{name}/rmsd_centers_and_k_for_imus.pkl", 'rb') as file_handle:
        data = pickle.load(file_handle)
        rmsd_centers_and_k[name] = data
sizes = [rmsd_centers_and_k[name]['size'] for name in protein_names]
cumsum_sizes = np.cumsum(sizes)

for idx_name in range(1, 4):
    for rank in range(cumsum_sizes[idx_name -1], cumsum_sizes[idx_name]):
        name = protein_names[idx_name]
        new_rank = rank - cumsum_sizes[idx_name -1]
        
        old_name = f"./output/{name}/traj_imus/traj_{rank}.dcd"
        new_name = f"./output/{name}/traj_imus/traj_{new_rank}.dcd"

        os.rename(old_name, new_name)
        print(name, old_name, new_name)

