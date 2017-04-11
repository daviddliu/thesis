import pandas as pd
import csv
import scipy
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sb
import itertools
import subprocess
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')


class ClusteringObject(object):
    """
        Contains all relevant information about an assignment of clusterings to a group of VAFs across samples.
        """

    def __init__(self, clustering_type, output_dir, data_dir, sim_name):
        # Initialize
        self.clustering_type = clustering_type
        self.data_dir = data_dir
        self.sim_name = sim_name
        self.output_path = output_dir

        # Params
        self.num_SNVs = None
        self.coverage = 1000
        self.num_samples = None
        # self.true_num_clusters = None
        # self.get_params_from_path()

        # Data
        self.VAFs = None
        self.reads_matrix = None
        self.var_reads = None
        self.ref_reads = None
        self.read_input_data()

        # True values
        # self.true_cluster_assignments = None
        # self.true_VAFs = None
        # self.true_pooled_reads = []
        # self.read_true_values()
        # self.pool_reads(true_reads=True)

        # Learned values: self.perform_clustering()
        self.putative_num_clusters = None
        self.putative_cluster_assignments = None

        # Calculated values
        self.putative_pooled_reads = []
        self.ARI = None
        self.cluster_freq_error = None

        # Compare w Ancestree
        self.num_mutations_placed = None

    # def get_params_from_path(self):
    #     directory_info = self.data_dir.split("_")
    #     coverage = int(directory_info[1])
    #     nsample = int(directory_info[3])
    #     nsnv = int(directory_info[5])
    #     ncluster = int(directory_info[7])
    #     self.num_samples = nsample
    #     self.coverage = coverage
    #     self.num_SNVs = nsnv
    #     self.true_num_clusters = ncluster

    # def read_true_values(self):
    #     """
    #         On init, read in the true values.
    #         """
    #     true_file_name = self.data_dir + ".true"
    #     with open(true_file_name, 'r+') as f:
    #         content = f.readlines()
    #     n_samples = int(content[2])
    #     self.num_samples = n_samples
    #     # Index 3 has the number of SNVs
    #     n_SNVs = int(content[3])
    #     self.num_SNVs = n_SNVs
    #
    #     # Now get the true clusters. Reverse the content array.
    #
    #     clusters = map(lambda cluster_str: map(int, cluster_str.split(";")),
    #                    content[::-1][1].split(" "))
    #     # Convert to a list where each SNV has a cluster from 0 to n-1.
    #     true_cluster_assignments = [0] * n_SNVs
    #     for i, cluster in enumerate(clusters):
    #         for SNV in cluster:
    #             true_cluster_assignments[SNV] = i
    #
    #     true_VAFs = []
    #
    #     # Cross reference to get the cluster frequencies
    #     # Offset 3
    #     for i in range(1 + 3, n_samples + 1 + 3):
    #         sample_VAFs = map(float, content[i].split(" "))
    #         cluster_VAFs = set()
    #         for i, VAF in enumerate(sample_VAFs):
    #             cluster_VAFs.add((VAF, true_cluster_assignments[i]))
    #
    #         true_sample_VAFs = [0] * len(clusters)
    #         for cluster_VAF in cluster_VAFs:
    #             true_sample_VAFs[cluster_VAF[1]] = cluster_VAF[0]
    #
    #         true_VAFs.append(true_sample_VAFs)
    #
    #     self.true_VAFs = true_VAFs
    #     self.true_cluster_assignments = true_cluster_assignments

    def read_input_data(self):
        """
            On init, read the data file.
            """
        true_file_name = self.data_dir
        reads_matrix = pd.DataFrame.from_csv(
            true_file_name, sep="\t").as_matrix()
        var_reads = [scipy.transpose(reads_matrix[:, 1])]
        ref_reads = [scipy.transpose(reads_matrix[:, 0])]
        n_SNVs = len(var_reads[0])
        self.num_SNVs = n_SNVs
        # Initialize the VAF_matrix, so skip the first two columns
        VAF_matrix = scipy.transpose(reads_matrix[:, 1] / map(
            float, reads_matrix[:, 0]))
        # Iterate over columns minus the first two
        reads_matrix = reads_matrix.T[2:]
        n_samples = 0
        for i, column in enumerate(reads_matrix):
            # Skip every other column
            if i % 2 != 0:
                continue
            n_samples += 1
            var_reads.append(reads_matrix[i + 1])
            ref_reads.append(reads_matrix[i])
            VAF_matrix = scipy.vstack((VAF_matrix, reads_matrix[i + 1] / map(
                float, reads_matrix[i])))

        self.num_samples = n_samples
        var_reads = map(scipy.ndarray.tolist, var_reads)
        ref_reads = map(scipy.ndarray.tolist, ref_reads)

        self.VAFs = VAF_matrix
        self.reads_matrix = reads_matrix
        self.var_reads = var_reads
        self.ref_reads = ref_reads
        # Data = bnpy.data.XData(X=scipy.transpose(scipy.matrix(VAF_matrix)))

    def perform_clustering(self):
        """
        This depends on the type of the clustering. Eg. BMM, Gaussian, etc. Must at the end call
        self.pool_reads(true_reads=False)
        self.calculate_performance()
        """
        raise NotImplementedError("Must implement a clustering method.")

    def pool_reads(self, true_reads=True):
        pooled_reads = []
        for sample_index in range(self.num_samples):
            cluster_params = []
            if true_reads:
                # for i in range(self.true_num_clusters):
                #     cluster_params.append([0, 0])
                # for i, (ref, var) in enumerate(zip(self.ref_reads[sample_index], self.var_reads[sample_index])):
                #     cluster_params[self.true_cluster_assignments[i]][0] += ref
                #     cluster_params[self.true_cluster_assignments[i]][1] += var
                # pooled_reads.append(cluster_params)
                raise Exception("There is no ground truth.")
            else:
                for i in range(self.putative_num_clusters):
                    cluster_params.append([0, 0])
                for i, (ref, var) in enumerate(zip(self.ref_reads[sample_index], self.var_reads[sample_index])):
                    cluster_params[self.putative_cluster_assignments[i]][0] += ref
                    cluster_params[self.putative_cluster_assignments[i]][1] += var
                pooled_reads.append(cluster_params)
        if true_reads:
            # self.true_pooled_reads = pooled_reads
            raise Exception("There is no ground truth.")
        else:
            self.putative_pooled_reads = pooled_reads

    # def calculate_performance(self):
    #     """
    #     Calculates the ARI and cluster frequency error of a clustering.
    #     """
    #     self.ARI = adjusted_rand_score(self.putative_cluster_assignments, self.true_cluster_assignments)
    #     # Cluster freq error
    #     cluster_freq_vector = []
    #     for sample_index in range(self.num_samples):
    #         for true_VAF in self.true_VAFs[sample_index]:
    #             cluster_freq_vector.append(min([abs((sr[1] / float(sr[0] + sr[1]) - true_VAF)) for sr in
    #                                             self.putative_pooled_reads[sample_index]]))
    #     self.cluster_freq_error = scipy.mean(cluster_freq_vector)

    # def create_ancestree_input(self, path):
    #     """
    #     Create the input file for ancestree, that reads thing.
    #     gene_id sample0 sample0 ... sample n sample n
    #     mut_1
    #     ...
    #     """
    #     with open(path, 'w+') as ancestree_input_file:
    #         writer = csv.writer(ancestree_input_file, delimiter='\t')
    #         # Header
    #         header = ['gene_id']
    #         for sample_id in range(self.num_samples):
    #             header.append("Sample_%d" % sample_id)
    #             header.append("Sample_%d" % sample_id)
    #         writer.writerow(header)
    #
    #         # Data
    #         for cluster_index in range(self.putative_num_clusters):
    #             gene_string = ""
    #             for gene_id, cluster_assgn in enumerate(self.putative_cluster_assignments):
    #                 if cluster_assgn == cluster_index:
    #                     gene_string += ";%d" % gene_id
    #             gene_string = gene_string[1:]
    #             row = [gene_string]
    #             for sample_index in range(self.num_samples):
    #                 ref = self.putative_pooled_reads[sample_index][cluster_index][0]
    #                 var = self.putative_pooled_reads[sample_index][cluster_index][1]
    #                 row.append(ref)
    #                 row.append(var)
    #             writer.writerow(row)

    # def evaluate_on_ancestree(self):
    #     """
    #     See how many mutations get placed by Ancestree.
    #     """
    #     ancestree_input_path = self.output_path + "/ancestree-input"
    #     ancestree_results_path = self.output_path + "/ancestree-results"
    #     self.create_ancestree_input(ancestree_input_path)
    #     with open(ancestree_results_path, 'w+') as ancestree_results_file:
    #         subprocess.call(['./ancestree', ancestree_input_path], stdout=ancestree_results_file)
    #
    #     cluster_sets = []
    #     mutations_placed = []
    #     with open(ancestree_results_path, 'r+') as ancestree_results_file:
    #         # Reverse the lines
    #         lines = ancestree_results_file.readlines()[::-1]
    #         for line in lines:
    #             if ";" in line:
    #                 # This is the semi-colon separated numbers
    #                 line = line.split('\t')[0]
    #                 mutations = line.split(";")
    #                 mutations_placed.append(mutations)
    #
    #     self.num_mutations_placed = len(list(itertools.chain.from_iterable(mutations_placed)))

    def get_cluster_plot_params(self, sample_index):
        """
        Pool the reads according to the cluster assignments. Add 1 for the prior.
        """
        raise NotImplementedError("You gotta super this bro, cos it depends on the type of model, don't it?")

    def generate_indiv_plot(self):
        """
        Generate an individual plot for this clustering.
        """
        with PdfPages(self.output_path + '/plot.pdf') as pdf:
            fig = plt.figure(figsize=(8, max(6, 2 * self.num_samples)))
            # Assign each cluster a color
            individual_VAF_colors = cm.rainbow(scipy.linspace(0, 1, self.putative_num_clusters))
            for sample_index in range(self.num_samples):
                # Plot the actual VAFs
                ax1 = fig.add_subplot(self.num_samples, 1, sample_index + 1)
                ax1.set_title("Sample %d" % sample_index, y=0.80)
                ax1.set_xlabel("VAF")
                ax1.set_ylabel("PDF")

                # if self.coverage > 500:
                #     ylim = self.coverage / 2
                # else:
                #     ylim = self.coverage
                # ax1.set_ylim(0, ylim)

                # Individual stuff
                for i, (ref_read, var_read) in enumerate(
                        zip(self.ref_reads[sample_index], self.var_reads[sample_index])):
                    a = var_read
                    b = ref_read
                    x = scipy.linspace(scipy.stats.beta.ppf(0.01, a, b), scipy.stats.beta.ppf(0.99, a, b), 100)
                    # Get the colors based on the cluster membership.
                    color = individual_VAF_colors[self.putative_cluster_assignments[i]]
                    ax1.plot(x, scipy.stats.beta.pdf(x, a, b), color=color, linewidth=0.5)

                # True clusters
                # plotted = dict()
                # for i, VAF in enumerate(self.true_VAFs[sample_index]):
                #     color = individual_VAF_colors[i]
                #     if VAF in plotted:
                #         plotted[VAF] -= 0.3
                #     else:
                #         plotted[VAF] = 2.0
                #     ax1.plot((VAF, VAF), (0, ylim), color=color, linewidth=plotted[VAF])

                # Overlay the clusters
                iteration = 0
                # Note that the method already makes 0 var and 1 ref.
                for params in self.get_cluster_plot_params(sample_index):
                    a = params[0]
                    b = params[1]
                    if a < 1e-12 and b < 1e-12:
                        continue
                    x = scipy.linspace(scipy.stats.beta.ppf(0.01, a, b), scipy.stats.beta.ppf(0.99, a, b), 100)
                    ax1.plot(x, scipy.stats.beta.pdf(x, a, b), 'black')
                    iteration += 1
                    # plt.rc('text', usetex=True)
                    # plt.rc('font', family='serif')

                fig.suptitle(
                    "%s | %s | %d clusters" % (
                    self.clustering_type, self.sim_name, iteration))
                if sample_index == 0:
                    true_patch = mpatches.Patch(color='white', label='True cluster frequencies and reads')
                    black_patch = mpatches.Patch(color='black', label='Cluster posterior distributions')
                    # brown_patch = mpatches.Patch(color='brown', label='Pooled reads beta distribution')

                    plt.legend(handles=[true_patch, black_patch], prop={'size': 6})

            # Show the plot
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            pdf.savefig()
            plt.close()
