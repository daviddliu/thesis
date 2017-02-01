from clustering_object import ClusteringObject
from CAVI import MultiBinomCAVI
import numpy as np

class MultiBinomMixtureModel(ClusteringObject):
    """
    Performs clustering on a model with multivariate binomial distributions under a DP allocation, with coordinate ascent variational inference.
    """

    def __init__(self, output_dir, data_dir, sim_name, kmeans=False, cvg_threshold=1.0, K=None):
        super(MultiBinomMixtureModel, self).__init__("MultiBinom", output_dir, data_dir, sim_name)
        # Default K is the number of data points
        if not K:
            self.initial_num_clusters = len(self.var_reads[0])//2
        else:
            self.initial_num_clusters = K
        self.kmeans = kmeans
        self.cvg_threshold = cvg_threshold

        self.clustering_time = None

        self.alpha_post = None
        self.beta_post = None
        self.num_clusters = None
        self.cluster_assgns = None
        self.cluster_params = None

        self.CAVI = MultiBinomCAVI(self.var_reads, self.ref_reads, kmeans=self.kmeans, cvg_threshold=self.cvg_threshold,
                                   K=self.initial_num_clusters)

    def perform_clustering(self):
        """
        Use CAVI to perform clustering.
        """
        import time
        start = time.time()
        self.CAVI.run()
        end = time.time()
        self.clustering_time = end - start
        print "Time elapsed: %f s" % self.clustering_time

        # Now set the posterior learned values.
        self.alpha_post = self.CAVI.variational_model.alpha
        self.beta_post = self.CAVI.variational_model.beta
        self.num_clusters = self.putative_num_clusters = self.CAVI.num_clusters
        self.putative_cluster_assignments = map(int, (self.CAVI.cluster_assgns - 1).tolist())
        self.cluster_params = self.CAVI.cluster_params

        return

    def get_cluster_plot_params(self, sample_index):
        """
        [[alpha, beta] for all clusters] ~ [var, ref]
        :param sample_index:
        :return:
        """
        return [[alphamk, betamk] for alphamk, betamk in zip(self.alpha_post[sample_index, :], self.beta_post[sample_index, :])]

    def write_output(self):
        """
        Write output to the data_dir. File name: clustering_output
        """
        from os import remove
        from os.path import exists
        out_file_loc = self.output_path + "/clustering_output"
        if exists(out_file_loc):
            remove(out_file_loc)
        with open(out_file_loc, "a+") as out_file:
            # Write parameters: convergence threshold, initial num clusters, kmeans
            out_file.write("MultiBinom clustering\n")
            out_file.write("Convergence threshold: %f\n" % self.cvg_threshold)
            out_file.write("Initial num clusters: %d\n" % self.initial_num_clusters)
            out_file.write("kmeans: %r\n\n" % self.kmeans)

            # Write time taken
            out_file.write("Time taken: %f\n" % self.clustering_time)
            # Write ARI
            out_file.write("ARI: %f\n" % self.ARI)
            # Write cluster freq error
            out_file.write("Cluster freq error: %f\n" % self.cluster_freq_error)


            # Write number clusters
            out_file.write("Num active clusters: %d\n" % self.putative_num_clusters)
            # Write cluster assignments
            out_file.write("Cluster assignments\n")
            np.savetxt(out_file, np.asarray(self.putative_cluster_assignments).T, fmt="%d")
            # Write alpha matrix
            out_file.write("Alpha posterior values\n")
            np.savetxt(out_file, self.alpha_post, fmt="%.4f")
            # Write beta matrix
            out_file.write("Beta posterior values\n")
            np.savetxt(out_file, self.beta_post, fmt="%.4f")
            # Write MAP phi matrix
            out_file.write("Cluster phi posterior values\n")
            np.savetxt(out_file, self.cluster_params, fmt="%.4f")