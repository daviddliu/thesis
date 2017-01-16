from clustering_object import ClusteringObject
from CAVI import MultiBinomCAVI

class MultiBinomMixtureModel(ClusteringObject):
    """
    Performs clustering on a model with multivariate binomial distributions under a DP allocation, with coordinate ascent variational inference.
    """

    def __init__(self, output_dir, data_dir, sim_name):
        super(MultiBinomMixtureModel, self).__init__("MultiBinom", output_dir, data_dir, sim_name)
        self.CAVI = MultiBinomCAVI(self.var_reads, self.ref_reads)
        self.alpha_post = None
        self.beta_post = None
        self.num_clusters = None
        self.cluster_assgns = None
        self.cluster_params = None

    def perform_clustering(self):
        """
        Use CAVI to perform clustering.
        """
        self.CAVI.run()

        # Now set the posterior learned values.
        self.alpha_post = self.CAVI.variational_model.alpha
        self.beta_post = self.CAVI.variational_model.beta
        self.num_clusters = self.CAVI.num_clusters
        self.cluster_assgns = self.CAVI.cluster_assgns
        self.cluster_params = self.CAVI.cluster_params