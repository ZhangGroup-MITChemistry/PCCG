import matplotlib.pyplot as plt
import numpy as np

loss_record = []
with open("./slurm_output/CL.txt", 'r') as file:
    flag = 0
    for line in file.readlines():
        line = line.strip()
        if 'X0' in line:
            flag += 1
        if flag == 2:
            if line[0:4] == "At i":
                loss = line[23:34].replace('D', 'E')
                loss_record.append(float(loss))

loss = np.array(loss_record)

fig = plt.figure()
fig.clf()
plt.plot(range(1, len(loss)+1), loss)
plt.xscale('log')
plt.xlabel('Number of iterations')
plt.ylabel('Aggregated loss with weight decay')
plt.savefig('./output/figures/loss.eps')
