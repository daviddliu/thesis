import itertools
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from MultiBinomMixtureModel import MultiBinomMixtureModel


class ClusteringModule(object):
    """
        Peforms clustering on multiple clustering objects, and provides summary statistics.
        """

    def __init__(self, clustering_type, data_dir="data/simulated/", kmeans=False):
        self.kmeans = kmeans
        self.clustering_type = clustering_type
        if kmeans:
            self.output_dir = "%s_results/" % (self.clustering_type + "_kmeans_k_175")
        else:
            self.output_dir = "%s_results/" % self.clustering_type
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.data_dir = data_dir
        self.df = None

    def run_on_all(self):
        directories = [directory for directory in os.listdir(self.data_dir)]
        file_names = ['sim_%d' % num for num in range(10)]
        # file_names = ["sim_9"]
        data_dicts = []

        for (info_dir, sim_num) in itertools.product(directories, file_names):
            if not os.path.exists(self.output_dir + info_dir):
                os.mkdir(self.output_dir + info_dir)
            data_path = self.data_dir + info_dir + '/' + sim_num
            results_dir = self.output_dir + info_dir + "/" + sim_num
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)

            if self.clustering_type == "MultiBinom":
                clustering_object = MultiBinomMixtureModel(results_dir, data_path, sim_num, kmeans=self.kmeans)
            else:
                raise Exception("Invalid clustering type specified. Your only option is MultiBinom.")

            clustering_object.perform_clustering()
            clustering_object.pool_reads(true_reads=False)
            clustering_object.calculate_performance()
            clustering_object.write_output()
            clustering_object.generate_indiv_plot()
            clustering_object.evaluate_on_ancestree()

            data_dicts.append({
                "num_SNVs": clustering_object.num_SNVs,
                "num samples": clustering_object.num_samples,
                "coverage": clustering_object.coverage,
                "clusters": clustering_object.putative_num_clusters,
                "ARI": clustering_object.ARI,
                "Cluster frequency error":
                    clustering_object.cluster_freq_error,
                "Num mutations placed":
                    clustering_object.num_mutations_placed
            })

        self.df = pd.DataFrame(data_dicts)

    def generate_violin_plots(self):

        clusters = sorted(list(set(self.df['clusters'].tolist())))
        coverages = sorted(list(set(self.df['coverage'].tolist())))
        samples = sorted(list(set(self.df['num samples'].tolist())))

        with PdfPages(self.output_dir + "%s_violin_plots.pdf" %
                self.clustering_type) as pdf:
            # Make violin plots
            fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
            axes.set_title(r"%s | Cluster frequency error" %
                           (self.clustering_type))
            axes.set_ylim(
                [0, self.df.max(axis=0)['Cluster frequency error'] + 0.01])
            s = sns.violinplot(
                x="coverage",
                y="Cluster frequency error",
                hue="num samples",
                order=coverages,
                hue_order=samples,
                scale="width",
                data=self.df,
                ax=axes,
                show_boxplot=False,
                cut=0,
                inner='point')
            plt.subplots_adjust(bottom=0.15)
            pdf.savefig()
            plt.close()

            fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
            axes.set_title(r"%s | ARI" % (self.clustering_type))
            axes.set_ylim([0, 1])
            s = sns.violinplot(
                x="coverage",
                y="ARI",
                hue="num samples",
                order=coverages,
                hue_order=samples,
                scale="width",
                data=self.df,
                ax=axes,
                show_boxplot=False,
                cut=0,
                inner='point')
            plt.subplots_adjust(bottom=0.15)
            pdf.savefig()
            plt.close()

            fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
            axes.set_title(r"%s | Number of clusters" %
                           (self.clustering_type))
            axes.set_ylim([0, self.df.max(axis=0)['clusters'] + 1])
            s = sns.violinplot(
                x="coverage",
                y="clusters",
                hue="num samples",
                order=coverages,
                hue_order=samples,
                scale="width",
                data=self.df,
                ax=axes,
                show_boxplot=False,
                cut=0,
                inner='point')
            plt.subplots_adjust(bottom=0.15)
            pdf.savefig()
            plt.close()

            fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
            axes.set_title(r"%s | Number of SNVs placed by Ancestree" %
                           (self.clustering_type))
            axes.set_ylim([0, self.df.max(axis=0)['num_SNVs'] + 1])
            s = sns.violinplot(
                x="coverage",
                y="Num mutations placed",
                hue="num samples",
                order=coverages,
                hue_order=samples,
                scale="width",
                data=self.df,
                ax=axes,
                show_boxplot=False,
                cut=0,
                inner='point')
            plt.subplots_adjust(bottom=0.15)
            pdf.savefig()
            plt.close()
