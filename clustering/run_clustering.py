from MultiBinomMixtureModel import *
from CAVI import MultiBinomCAVI
import ipdb

"""
Single run code.
"""

def run_single(data_dir="data/simulated/Cov_1000_Samples_4_Mut_100_Clone_10_PCR_Removed/sim_0", sim="sim_0"):
    clustering_model = MultiBinomMixtureModel("single_output", data_dir, sim)
    clustering_model.perform_clustering()

    # Put learned values into the model.
    clustering_model.putative_num_clusters = clustering_model.CAVI.num_clusters
    clustering_model.putative_cluster_assignments = map(int, (clustering_model.CAVI.cluster_assgns - 1).tolist())
    clustering_model.pool_reads(true_reads=False)

    # Make the plots.
    clustering_model.calculate_performance()
    print "ARI: %f" % clustering_model.ARI
    print "Number clusters: %d" % clustering_model.CAVI.num_clusters
    clustering_model.generate_indiv_plot()

"""
Run on all data code.
"""
def run_all():
