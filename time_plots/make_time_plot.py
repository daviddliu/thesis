import os
from collections import defaultdict
import itertools

def get_DPVI_times():
    times = defaultdict(list)
    data_dir = "MultiBinom_kmeans_k_100_results"
    directories = [directory for directory in os.listdir(data_dir)]
    file_names = ['sim_%d' % num for num in range(10)]

    for (info_dir, sim_num) in itertools.product(directories, file_names):
        file_loc = data_dir + '/' + info_dir + '/' + sim_num + '/' + 'clustering_output'
        with open(file_loc, 'r+') as info_file:
            time_str = info_file.readlines()[5]
            time = float(time_str.split(": ")[1].split("\n")[0])
            num_samples = int(info_dir.split("_")[3])
            times[num_samples].append(time)
    return times


def get_SciClone_times():
    times = defaultdict(list)
    with open('SciClonetimes.txt', 'r+') as times_file:
        times_lines = times_file.readlines()
        for times_line in times_lines:
            times_str = times_line.split("\t")
            num_samples = int(times_str[0])
            time = float(times_str[1].split('\n')[0])
            times[num_samples].append(time)
    return times

def get_PyClone_times():
    times = defaultdict(list)
    with open('PyClonetimes.txt', 'r+') as times_file:
        times_lines = times_file.readlines()
        for times_line in times_lines:
            times_str = times_line.split(",")
            num_samples = int(times_str[1].split('\r')[0])
            time = float(times_str[0])
            times[num_samples].append(time)
    return times

DPVI_times = get_DPVI_times()
SciClone_times = get_SciClone_times()
PyClone_times = get_PyClone_times()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Make a pandas dataframe for the plot
data_dicts = []
for num_samples, times in DPVI_times.iteritems():
    for time in times:
        data_dicts.append({
            "Number of samples": num_samples,
            "Convergence time (s)": time,
            "Inference method": "DP/VI"
            })
for num_samples, times in SciClone_times.iteritems():
    for time in times:
        data_dicts.append({
            "Number of samples": num_samples,
            "Convergence time (s)": time,
            "Inference method": "SciClone (VI)"
            })
for num_samples, times in PyClone_times.iteritems():
    for time in times:
        data_dicts.append({
            "Number of samples": num_samples,
            "Convergence time (s)": time,
            "Inference method": "PyClone (DP/MCMC)"
            })

data = pd.DataFrame(data_dicts)
ax = sns.pointplot(x="Number of samples", y="Convergence time (s)", hue="Inference method", data=data, capsize=0.1, markers=['x', 'o', '^'], linestyles=['--', '--', '--'])
sns.plt.savefig('time_comparisons.png')
