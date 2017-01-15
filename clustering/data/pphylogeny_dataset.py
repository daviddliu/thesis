#!/bin/python

import bnpy as bnpy
import scipy as scipy
import scipy.stats
from collections import defaultdict
import pandas as pd

def get_short_name():
    return 'phylogeny_dataset'

def read_true_file(true_file_name):
    with open(true_file_name, 'r+') as f:
        content = f.readlines()
    n_samples = int(content[2])
    # Index 3 has the number of SNVs
    n_SNVs = int(content[3])
    
    # Now get the true clusters. Reverse the content array.

    clusters = map(lambda cluster_str: map(int, cluster_str.split(";")), content[::-1][1].split(" "))
    # Convert to a list where each SNV has a cluster from 0 to n-1.
    true_cluster_assignments = [0] * n_SNVs 
    for i, cluster in enumerate(clusters):
        for SNV in cluster:
            true_cluster_assignments[SNV] = i
    
    true_VAFs = []

    # Cross reference to get the cluster frequencies
    # Offset 3
    for i in range(1+3, n_samples+1+3):
        sample_VAFs = map(float, content[i].split(" "))
        cluster_VAFs = set()
        for i, VAF in enumerate(sample_VAFs):
            cluster_VAFs.add((VAF, true_cluster_assignments[i]))

        true_sample_VAFs = [0] * len(clusters)
        for cluster_VAF in cluster_VAFs:
            true_sample_VAFs[cluster_VAF[1]] = cluster_VAF[0]

        true_VAFs.append(true_sample_VAFs)

    return true_VAFs, true_cluster_assignments


def get_data(**kwargs):
    data_file_name = kwargs['file_name'] + '.input'
    true_file_name = kwargs['file_name'] + '.true'

    reads_matrix = pd.DataFrame.from_csv(data_file_name, sep="\t").as_matrix()
    var_reads = [scipy.transpose(reads_matrix[:,1])]
    ref_reads = [scipy.transpose(reads_matrix[:,0])]
    # Initialize the VAF_matrix, so skip the first two columns
    VAF_matrix = scipy.transpose( reads_matrix[:,1]/ map(float, reads_matrix[:,0]))
    # Iterate over columns minus the first two
    reads_matrix = reads_matrix.T[2:]
    for i, column in enumerate(reads_matrix):
        # Skip every other column
        if i % 2 != 0:
            continue
        var_reads.append(reads_matrix[i+1])
        ref_reads.append(reads_matrix[i])
        VAF_matrix = scipy.vstack((VAF_matrix, reads_matrix[i+1]/map(float, reads_matrix[i])))

    var_reads = map(scipy.ndarray.tolist, var_reads)
    ref_reads = map(scipy.ndarray.tolist, ref_reads)

    true_VAFs, true_clusters = read_true_file(true_file_name)
    Data = bnpy.data.XData(X=scipy.transpose(scipy.matrix(VAF_matrix)))

    return (Data, var_reads, ref_reads, true_clusters, true_VAFs)

