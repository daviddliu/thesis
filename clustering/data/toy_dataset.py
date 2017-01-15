#!/bin/python

import bnpy as bnpy
import scipy as scipy
import scipy.stats


def get_short_name():
    return 'toy_dataset'

def get_data(**kwargs):
    # Generate a list of var/ref counts. 
    scipy.random.seed(seed=kwargs['seed'])
    # Number of SNV sites
    N = kwargs['n']
    # Number of samples
    P = kwargs['p']
    read_depth = kwargs['r']
    # Number of different binomial things
    true_num_clusters = kwargs['c']
    binom_probs = [scipy.random.uniform(0, 1.0, true_num_clusters) for sample in range(P)]
    read_depths = [scipy.stats.poisson.rvs(read_depth, size=N) for sample in range(P)]
    # Remember which binomials were chosen
    var_reads = []
    # Assign mutations to clusters
    true_clusters = [scipy.random.randint(true_num_clusters) for i in range(N)]
    
    # Generate values for each SNV in each cluster for each sample
    for p, sample_read_depths in enumerate(read_depths):
        sample_var_reads = []
        for n, read_depth in enumerate(sample_read_depths):
            true_cluster = true_clusters[n]
            sample_var_reads.append(scipy.stats.binom.rvs(read_depth, binom_probs[p][true_cluster]))
        var_reads.append(sample_var_reads)
    
    # Get VAFs
    import operator
    ref_reads = []
    VAFs = []
    for (read_depth_arr, var_read_arr) in zip(read_depths, var_reads):
        VAFs.append(map(operator.div, var_read_arr, map(float, read_depth_arr)))
        ref_reads.append(map(operator.sub, read_depth_arr, var_read_arr))
    
    Data = bnpy.data.XData(X=scipy.transpose(scipy.matrix(VAFs)))
    
    #Data = convertCountMatToWordsData(count_mat)
    Data.name = get_short_name()
    return (Data, true_clusters, var_reads, ref_reads, binom_probs)

