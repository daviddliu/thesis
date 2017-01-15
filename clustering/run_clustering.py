from MultiBinomMixtureModel import *
from CAVI import MultiBinomCAVI
import ipdb

clustering_model = MultiBinomMixtureModel("output/", "data/simulated/Cov_1000_Samples_4_Mut_100_Clone_10_PCR_Removed/sim_0", "sim_0")
CAVI = MultiBinomCAVI(clustering_model.var_reads, clustering_model.ref_reads)
CAVI.run()

# Put learned values into the model.
clustering_model.putative_num_clusters = CAVI.num_clusters
clustering_model.putative_cluster_assignments = map(int, (CAVI.cluster_assgns - 1).tolist())
clustering_model.pool_reads(true_reads=False)

# Make the plots.
clustering_model.calculate_performance()
print "ARI: %f" % clustering_model.ARI
print "Number clusters: %d" % CAVI.num_clusters
clustering_model.generate_indiv_plot()