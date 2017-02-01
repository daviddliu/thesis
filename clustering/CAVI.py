import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import entropy
from util import log_choose, inplaceExpAndNormalizeRows, calcBetaExpectations
from sklearn.cluster import KMeans
import ipdb

class VariationalModel(object):
    """
    Observation model: Multivariate Binomial

    Prior
    -----
    alpha0: M X K array
        Flat prior, all 1's
    beta0: M X K array
        Flat prior, all 1's

    Params
    ------
    phi: M X K array
        Each column (cluster) has the M emission frequencies.

    Posterior
    ---------
    alpha: M X K array
    beta: M X K array

    Sufficient Statistics
    ---------------------
    N: N X K array
        Passed through the allocation model.
    xvar: M X K array
        Each column (cluster) has the M estimated variant reads per sample.
    xref: M X K array
        Each column (cluster) has the M estimated reference reads per sample.



    Allocation model: Dirichlet Process

    Prior
    -----
    gamma1: float, defaults to 1.0
    gamma0: float, default to 5.0

    Params
    ------
    K: int
        Number of clusters.
    Z: 1D array, size N
        Hard assignments of data to clusters.

    Posterior
    ---------
    eta1: 1D array, size K
        ON for Beta post
    eta0: 1D array, size K
        OFF for Beta post
    r: 2D array, size N X K
        Array of cluster responsibilities.

    """
    def __init__(self, var_reads, ref_reads, K, kmeans=False):
        # Data
        self.var_reads = var_reads
        self.ref_reads = ref_reads

        # -------------------------------#
        # Parameters for the alloc model #
        # -------------------------------#
        ## Global params
        self.N = self.var_reads.shape[1]
        self.M = self.var_reads.shape[0]
        self.K = K

        # Sufficient statistics
        # Initialize to equal responsibilities.
        self.r = (1.0/K) * np.ones((self.N, self.K))
        self.Nk = np.ones(self.K)
        self.Nk_gt = np.ones(self.K)

        ## Prior
        # TODO: Not sure why. But it is this in bnpy. I think gamma0 is alpha.
        self.gamma1 = 1.0
        self.gamma0 = 1.5
        ## Posterior
        self.eta1 = np.full(K, self.gamma1)
        self.eta0 = np.full(K, self.gamma0)
        for k in xrange(K):
            self.eta1[k] += (float(k)/K)
            self.eta0[k] += (1 - float(k)/K)

        # ------------------------------#
        #  Parameters for the obs model #
        # ------------------------------#
        ## Prior
        self.alpha0 = np.ones(var_reads.shape)
        self.beta0 = np.ones(ref_reads.shape)
        ## Params
        # self.phi
        ## Posterior: initialize to the prior values.
        self.alpha = np.ones(var_reads.shape)
        self.beta = np.ones(ref_reads.shape)
        # Sufficient statistics
        # Gets initialized when calculating suff stats.
        self.Sxvar = np.zeros(var_reads.shape)
        self.Sxref = np.zeros(ref_reads.shape)

        self.Elog_data_likelihood = np.empty((self.N, self.K))
        self.Elog_stick_likelihood = np.empty((self.N, self.K))
        self.weights = np.empty((self.N, self.K))

        self.ELBO = None

        if kmeans:
            self.VAFs = np.true_divide(self.var_reads, self.var_reads + self.ref_reads)
            kmeans = KMeans(n_clusters=K / 2, random_state=0).fit(self.VAFs.T)
            khard_assgns = kmeans.labels_
            # Initialize r
            for i, label in enumerate(khard_assgns):
                self.r[i][label] = 1

            # Initialize suff stats
            self.calc_suff_stats()

            # Initialize global params
            self.calc_global_params()

            # Initialize local params
            self.calc_local_params()

    def calc_local_params(self):
        # ObsModel: Calculate the variational likelihood matrix E[ln p(x_n | a_k, b_k)]
        for n in xrange(self.N):
            for k in xrange(self.K):
                elog_likelihood = 0
                for m in xrange(self.M):
                    var_reads = self.var_reads[m][n]
                    ref_reads = self.ref_reads[m][n]
                    total_reads = var_reads + ref_reads
                    elog_likelihood += (log_choose(total_reads, var_reads)
                                        + var_reads * (digamma(self.alpha[m][k]) - digamma(self.alpha[m][k] + self.beta[m][k]))
                                        + ref_reads * (digamma(self.beta[m][k]) - digamma(self.alpha[m][k] + self.beta[m][k])))
                self.Elog_data_likelihood[n][k] = elog_likelihood
        # AllocModel
        ## Calculate weights
        ElogU, Elog1mU = calcBetaExpectations(self.eta1, self.eta0)

        # Calculate expected mixture weights E[ log \beta_k ]
        # Using copy() allows += without modifying ElogU
        self.Elog_stick_likelihood = ElogU.copy()
        self.Elog_stick_likelihood[1:] += Elog1mU[:-1].cumsum()
        self.weights = self.Elog_stick_likelihood + self.Elog_data_likelihood

        ## Calculate responsibilities
        # TODO: Check for numerical stability
        self.r = self.weights.copy()
        inplaceExpAndNormalizeRows(self.r)
        # print "Calculated local params."
        return

    def calc_suff_stats(self):
        # Update the read counts for each cluster
        for k in xrange(self.K):
            var_weighted_sum_k = np.zeros((self.M))
            ref_weighted_sum_k = np.zeros((self.M))
            for n in xrange(self.N):
                var_weighted_sum_k += self.r[n][k] * self.var_reads[:, n]
                ref_weighted_sum_k += self.r[n][k] * self.ref_reads[:, n]
            self.Sxvar[:, k] = var_weighted_sum_k
            self.Sxref[:, k] = ref_weighted_sum_k

        # Update Nk
        self.Nk = np.sum(self.r, axis=0)
        # Update Nk >
        total_Nk = np.sum(self.Nk[1:])
        for k in xrange(self.K - 1):
            self.Nk_gt[k] = total_Nk
            total_Nk -= self.Nk[k+1]
        self.Nk_gt[self.K - 1] = 0

        # print "Calculated suff stats."
        return

    def calc_global_params(self):
        # ObsModel
        self.alpha = (self.alpha0 - 1) + self.Sxvar
        self.beta = (self.beta0 - 1) + self.Sxref
        self.eta1 = self.gamma1 + self.Nk
        self.eta0 = self.gamma0 + self.Nk_gt

        # print "Calculated global params."
        return


    def calc_ELBO(self):
        # L_obs
        E_ln_data_likelihood = 0
        for n in xrange(self.N):
            for k in xrange(self.K):
                E_ln_data_likelihood += self.r[n][k] * self.Elog_data_likelihood[n][k]

        ## The flat prior means this contributes nothing.
        E_ln_prior = 0

        E_q_ln_prior = 0
        for k in xrange(self.K):
            for m in xrange(self.M):
                alpha = self.alpha[m][k]
                beta = self.beta[m][k]
                E_q_ln_prior += (gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
                                    + (alpha - 1) * (digamma(alpha) - digamma(alpha + beta))
                                    + (beta - 1) * (digamma(beta) - digamma(alpha + beta)))

        L_obs = E_ln_data_likelihood + E_ln_prior - np.log(E_q_ln_prior)
        print "L_obs: %f" % L_obs

        # L_alloc
        digamma_term = 0
        e_log_uk_term = 0
        e_log_m_uk_term = 0
        for k in xrange(self.K):
            digamma_term += (gammaln(1 + self.gamma0) - gammaln(1) - gammaln(self.gamma0)
                            - (gammaln(self.eta1[k] + self.eta0[k]) - gammaln(self.eta1[k]) - gammaln(self.eta0[k])))
            e_log_uk_term += (self.Nk[k] + 1 - self.eta1[k]) * (digamma(self.eta1[k]) - digamma(self.eta1[k] + self.eta0[k]))
            e_log_m_uk_term += (self.Nk_gt[k] + 1 - self.eta0[k]) * (digamma(self.eta0[k]) - digamma(self.eta1[k] + self.eta0[k]))

        L_alloc = digamma_term + e_log_uk_term + e_log_m_uk_term
        print "L_alloc: %f" % L_alloc

        # L_entropy
        L_entropy = entropy(self.r.flatten())
        print "L_entropy: %f" % L_entropy

        self.ELBO = L_obs + L_alloc + L_entropy

        return self.ELBO

    def convert_to_params(self):
        """
        For each 1,...,n data points, returns the index of the cluster it is assigned to, and the cluster parameters (MAP estimate).
        """
        # Hard cluster assignments
        cluster_assgns = np.empty(self.N)
        for n in xrange(self.N):
            cluster_assgns[n] = np.argmax(self.r[n, :])

        # MAP parameter estimates
        active_cluster_indices = set(cluster_assgns)
        cluster_params = {}
        for k in active_cluster_indices:
            k = int(k)
            putative_cluster_params = np.divide((self.Sxvar[:, k] / self.Nk[k]) + self.alpha[:, k] - 1,
                                        (self.Sxvar[:, k] + self.Sxref[:, k])/self.Nk[k] + self.alpha[:, k] + self.beta[:, k] - 2)
            # Some putative params may be negative. Make them positive.
            cluster_params[k] = map(lambda x: 0 if x < 0 else x, putative_cluster_params)

        # Reassign indices
        new_index = 1
        old_to_new_map = {}
        for k in set(cluster_assgns):
            old_to_new_map[k] = new_index
            new_index += 1

        new_cluster_assgns = np.empty(self.N)
        for i, k_old in enumerate(cluster_assgns):
            new_cluster_assgns[i] = old_to_new_map[k_old]

        new_cluster_params = np.empty((self.M, len(active_cluster_indices)))

        for k_old, param in cluster_params.items():
            new_cluster_params[:, old_to_new_map[k_old] - 1] = param

        return new_cluster_assgns, new_cluster_params, len(active_cluster_indices)


class MultiBinomCAVI(object):
    """
    CAVI for the MultiBinomMixtureModel, with DP allocation.
    """
    def __init__(self, var_reads, ref_reads, K=None, cvg_threshold=None, kmeans=None):
        # Data
        self.var_reads = np.asarray(var_reads)
        self.ref_reads = np.asarray(ref_reads)

        # Tuning parameters
        self.cvg_threshold = cvg_threshold
        self.K = K

        # Params
        self.cluster_assgns = None
        self.cluster_params = None
        self.num_clusters = None

        # Initialize the model
        self.variational_model = VariationalModel(self.var_reads, self.ref_reads, self.K, kmeans=kmeans)

    def run(self):
        print "Initializing..."
        prev_bound = -np.inf
        is_converged = False
        lap = 0

        while not is_converged:
            lap += 1
            # Calculate local parameters (data likelihoods and responsibilities)
            self.variational_model.calc_local_params()

            # Calculate sufficient statistics (xvar, xref, N)
            self.variational_model.calc_suff_stats()

            # Calculate global parameters (cluster parameters)
            self.variational_model.calc_global_params()

            # Calculate ELBO, test for convergence
            new_ELBO = self.variational_model.calc_ELBO()
            print "Finished lap %d | ELBO: %f" % (lap, new_ELBO)
            if abs(new_ELBO - prev_bound) <= self.cvg_threshold:
                is_converged = True
            prev_bound = new_ELBO

        # TODO: Don't forget to re-index the alpha, beta, cluster params.
        self.cluster_assgns, self.cluster_params, self.num_clusters = self.variational_model.convert_to_params()
        print "Finished CAVI."
        return