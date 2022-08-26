__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/12/14 02:42:51"

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
torch.set_default_dtype(torch.double)
import pickle
from functions import *
from sys import exit
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

with open("./output/range.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max = data['x1_min'], data['x1_max']
x2_min, x2_max = data['x2_min'], data['x2_max']

num_reps = 10
alphas = torch.linspace(0.0, alpha, num_reps)

num_steps = 5100000
x_record = []
accept_rate = 0
x = torch.stack((x1_min + torch.rand(num_reps)*(x1_max - x1_min),
                 x2_min + torch.rand(num_reps)*(x2_max - x2_min)),
                dim = -1)
energy = compute_Muller_potential(1.0, x)

for k in range(num_steps):
    if (k + 1) % 10000 == 0:
        print("idx of steps: {}".format(k))
        
    ## sampling within each replica
    delta_x = torch.normal(0, 1, size = (num_reps, 2))*0.3
    x_p = x + delta_x
    energy_p = compute_Muller_potential(1.0, x_p)

    ## accept based on energy
    accept_prop = torch.exp(-alphas*(energy_p - energy))
    accept_flag = torch.rand(num_reps) < accept_prop

    ## considering the bounding effects
    accept_flag = accept_flag & torch.all(x_p > x_p.new_tensor([x1_min, x2_min]), -1) \
                              & torch.all(x_p < x_p.new_tensor([x1_max, x2_max]), -1)
    
    x_p[~accept_flag] = x[~accept_flag]
    energy_p[~accept_flag] = energy[~accept_flag]    
    x = x_p
    energy = energy_p

    ## calculate overall accept rate
    accept_rate = accept_rate + (accept_flag.float() - accept_rate)/(k+1)    
    
    ## exchange
    if k % 10 == 0:
        for i in range(1, num_reps):
            accept_prop = torch.exp((alphas[i] - alphas[i-1])*(energy[i] - energy[i-1]))
            accept_flag = torch.rand(1) < accept_prop
            if accept_flag.item():
                tmp = x[i].clone()
                x[i] = x[i-1]
                x[i-1] = tmp

                tmp = energy[i].clone()
                energy[i] = energy[i-1]
                energy[i-1] = tmp
        if k >= 100000:
            x_record.append(x.clone().numpy())

x_record = np.array(x_record)
os.makedirs(f"./output/TREMC/", exist_ok = True)
with open("./output/TREMC/x_record_alpha_{:.3f}.pkl".format(alpha), 'wb') as file_handle:
    pickle.dump({'alphas': alphas,
                 'x_record': x_record}, file_handle)

samples_plots = PdfPages("./output/TREMC/samples_alpha_{:.3f}.pdf".format(alpha))
for i in range(num_reps):
    print(i)
    fig = plt.figure(0)
    fig.clf()
    plt.plot(x_record[:, i, 0], x_record[:, i, 1], '.', alpha = 0.5)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.title(r"$\alpha = {:.3f}$".format(alphas[i]))
    samples_plots.savefig()
    
samples_plots.close()
