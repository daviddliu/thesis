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

    def __init__(self, clustering_type, data_dir="data/real/", kmeans=False):
        self.kmeans = kmeans
        self.clustering_type = clustering_type
        # self.cancer_type = "Kidney"
        self.cancer_type = "Lung"
        # self.cancer_type = "CLL"
        if kmeans:
            self.output_dir = "real_%s_%s_results/" % (self.cancer_type, self.clustering_type + "_kmeans")
        else:
            self.output_dir = "real_%s_%s_results/" %(self.cancer_type,  self.clustering_type)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.data_dir = data_dir
        self.df = None

    def run_on_all(self):
        # Kidney
        # file_names = ["EV005.txt", "EV003.txt", "EV006.txt","EV007.txt","RK26.txt","RMH002.txt","RMH004.txt","RMH008.txt"]
        # Lung
        file_names = ["270_deep.txt","270_whole.txt","283_deep.txt","283_whole.txt","292_deep.txt","292_whole.txt",
                      "317_deep.txt","317_whole.txt","324_deep.txt","324_whole.txt","330_deep.txt","330_whole.txt",
                      "339_deep.txt","339_whole.txt","356_deep.txt","356_whole.txt","472_deep.txt","472_whole.txt",
                      "4990_deep.txt","4990_whole.txt","499_deep.txt","499_whole.txt"]
        # CLL
        # file_names = ["CLL006_whole.txt", "CLL003_deep.txt","CLL003_whole.txt","CLL006_deep.txt",
        #               "CLL077_deep.txt","CLL077_whole.txt"]
        data_dicts = []

        for file_name in file_names:
            file_descriptor = file_name.split(".")[0]
            if not os.path.exists(self.output_dir + file_descriptor):
                os.mkdir(self.output_dir + file_descriptor)
            data_path = self.data_dir + file_name
            results_dir = self.output_dir + file_descriptor
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)

            if self.clustering_type == "MultiBinom":
                clustering_object = MultiBinomMixtureModel(results_dir, data_path, file_descriptor,
                                                           kmeans=self.kmeans)
            else:
                raise Exception("Invalid clustering type specified. Your only option is MultiBinom.")

            clustering_object.perform_clustering()
            clustering_object.pool_reads(true_reads=False)
            # clustering_object.calculate_performance()
            clustering_object.write_output()
            clustering_object.generate_indiv_plot()
            # clustering_object.evaluate_on_ancestree()

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
                (self.clustering_type + self.cancer_type)) as pdf:
            # Make violin plots
            fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
            # axes.set_title(r"%s | Cluster frequency error" %
            #                (self.clustering_type))
            # axes.set_ylim(
            #     [0, self.df.max(axis=0)['Cluster frequency error'] + 0.01])
            # s = sns.violinplot(
            #     x="coverage",
            #     y="Cluster frequency error",
            #     hue="num samples",
            #     order=coverages,
            #     hue_order=samples,
            #     scale="width",
            #     data=self.df,
            #     ax=axes,
            #     show_boxplot=False,
            #     cut=0,
            #     inner='point')
            # plt.subplots_adjust(bottom=0.15)
            # pdf.savefig()
            # plt.close()

            # fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
            # axes.set_title(r"%s | ARI" % (self.clustering_type))
            # axes.set_ylim([0, 1])
            # s = sns.violinplot(
            #     x="coverage",
            #     y="ARI",
            #     hue="num samples",
            #     order=coverages,
            #     hue_order=samples,
            #     scale="width",
            #     data=self.df,
            #     ax=axes,
            #     show_boxplot=False,
            #     cut=0,
            #     inner='point')
            # plt.subplots_adjust(bottom=0.15)
            # pdf.savefig()
            # plt.close()

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
            #
            # fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
            # axes.set_title(r"%s | Number of SNVs placed by Ancestree" %
            #                (self.clustering_type))
            # axes.set_ylim([0, self.df.max(axis=0)['num_SNVs'] + 1])
            # s = sns.violinplot(
            #     x="coverage",
            #     y="Num mutations placed",
            #     hue="num samples",
            #     order=coverages,
            #     hue_order=samples,
            #     scale="width",
            #     data=self.df,
            #     ax=axes,
            #     show_boxplot=False,
            #     cut=0,
            #     inner='point')
            # plt.subplots_adjust(bottom=0.15)
            # pdf.savefig()
            # plt.close()
