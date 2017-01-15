from clustering_object import ClusteringObject
from CAVI import MultiBinomCAVI

class MultiBinomMixtureModel(ClusteringObject):
    """
    Performs clustering on a model with multivariate binomial distributions under a DP allocation, with coordinate ascent variational inference.
    """

    def __init__(self, output_dir, data_dir, sim_name):
        super(MultiBinomMixtureModel, self).__init__("MultiBinom", output_dir, data_dir, sim_name)

    def perform_clustering(self):
        """
        Use CAVI to perform clustering.
        """
        ref_reads = self.ref_reads
        var_reads = self.var_reads

        cluster_model = MultiBinomCAVI()
        cluster_model.run()
