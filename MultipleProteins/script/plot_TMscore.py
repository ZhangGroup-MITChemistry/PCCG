import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

score_md = pd.read_table('./output/test/TM_score_md.txt', header = None, sep = '[=(]')
score_md = score_md.iloc[:, 1].tolist()

score_cg = pd.read_table('./output/test/TM_score_cg.txt', header = None, sep = '[=(]')
score_cg = score_cg.iloc[:, 1].tolist()

fig = plt.figure()
fig.clf()
plt.hist(score_md, bins = 40, density = True, label = 'md', alpha = 0.5)
plt.hist(score_cg, bins = 40, density = True, label = 'cg', alpha = 0.5)
plt.savefig(f"./output/test/tmscore_hist.pdf")
