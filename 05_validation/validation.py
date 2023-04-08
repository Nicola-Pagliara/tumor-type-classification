import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from support import support
from collections import OrderedDict
from operator import itemgetter

def gene_confidence_score_heatmaps(out_path_base, classes):
    """
    Function to get and save the confidence score from Guided Grad-CAM images
    :param out_path_base: base path to save data
    :type out_path_base: str
    :param classes: list of classes names
    :type classes: list
    :return: None
    """
    for folder_no in range(0, support.NUM_FOLDERS):
        out_path = out_path_base + str(folder_no)
        for i in tqdm(range(0, len(classes))):
            # CAUTION ! Make sure to set the correct path for heatmaps in function transform_img_to_data in support.py
            D = support.transform_img_to_data(out_path_base, folder_no, classes[i])
            if not os.path.exists(os.path.join(out_path, 'ConfidenceScore')):
                os.makedirs(os.path.join(out_path, 'ConfidenceScore'))
            D.to_csv(os.path.join(out_path, 'ConfidenceScore', classes[i] + '_confidence.csv'), header=False, index=False)
        print('Complete to save the trasform img to data for folder:', folder_no)


def gen_plot(feature_path, img_dir_base, classes):
    """
    Function to generate plots
    :param feature_path: path of "feature_name.csv"
    :type feature_path: str
    :param img_dir_base: base path for output
    :type img_dir_base: str
    :param classes: list of classes names
    :type classes: list
    """
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # to color the plot
    num_genes = pd.DataFrame(pd.read_csv(feature_path, sep=',', header=None))
    count = int(str(num_genes[0].shape[0]))
    print('count: ', count)
    for folder_no in range(0, 10):
        np_array_num_genes = []

        for i in range(1, count):
            np_array_num_genes.append(num_genes.iloc[i][0].astype(int))

        print('np_array_numgenes: ', np_array_num_genes)
        dataframe_scores = []  # list of dataframes

        figure, axis = plt.subplots(len(classes), 1, sharex='all')
        # rank genes score extraction
        classes_arry = []  # all classes
        for i in range(0, len(classes)):
            dataframe_scores.append(pd.DataFrame(
                # pd.read_csv(img_dir_base + str(folder_no) + '/ConfidenceScore/' + classes[i] + '_confidence.csv', sep=',', header=None)))
                pd.read_csv(img_dir_base + str(folder_no) + '/Confidence_score_VarNet/' + classes[i] + '_confidence.csv', sep=',', header=None)))
            print('Confidence score for class: ', classes[i], dataframe_scores)
            class_arry = []  # single class
            for ii in range(0, int(str(num_genes[0].shape[0])) - 1):
                class_arry.append(dataframe_scores[i].iloc[ii][0])

            class_arry = np.sort(class_arry)
            class_arry = class_arry[::-1]
            classes_arry.append(class_arry)
        print('Complete to extract rank genes score for folder:', folder_no)

        # Plot 2D graphic which are plot the line represent the rank of genes.
        # Order in decescent way the rank for obtain a decrescent function
        print('END loop for confidence score')

        # following loop is useful to obtain a separate plot for each class under test
        # (used only in binary and ternary test, comment it in general test)
        # for cls in range(0, len(classes_arry)):
        #     axis[cls].set_title("# Tumor genes vs rank gene class " + classes[cls] + " Fold " + str(folder_no), \
        #                         fontdict={'fontsize': 10, \
        #                                   'fontweight': 'bold', \
        #                                   'color': 'black', \
        #                                   'verticalalignment': 'baseline', \
        #                                   'horizontalalignment': 'center'})
        #     axis[cls].plot(np_array_num_genes, classes_arry[cls], color=colors[cls % len(colors)], label=classes[cls])
        #     axis[cls].set_ylabel("rank genes")
        #     axis[cls].legend()
            # plt.plot(np_array_num_genes, classes_arry[cls], color=colors[cls % len(colors)], label=classes[cls])

        # path to save figures
        save_fig_path = img_dir_base + str(folder_no) + '/plot_VarNet'
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)

        # utils for deleting old pictures if rerun is needed:

        # if os.path.exists(save_fig_path + '/separate_plot.png'):
        #     os.remove(save_fig_path + '/separate_plot.png')
        # if os.path.exists(save_fig_path + '/general_plot.png'):
        #      os.remove(save_fig_path + '/general_plot.png')
        if os.path.exists(save_fig_path + '/gf_general_plot.png'):
            os.remove(save_fig_path + '/gf_general_plot.png')

        # plt.xlabel("# of tumor genes")
        # plt.savefig(os.path.join(save_fig_path, 'separate_plot'))
        # plt.close()

        # single figure (all classes on the same plot)
        plt.axis([0, 10410, 0, 250]) # axis dim (x = [0, 10410], y = [0, 250])
        for cls in range(0, len(classes_arry)):
            # plot lines smoothening
            ysmoothed = gaussian_filter(classes_arry[cls], sigma=100, truncate=1.1, mode='nearest')
            plt.plot(np_array_num_genes, ysmoothed, color=colors[cls % len(colors)], label=classes[cls])

        plt.xlabel('# of tumor genes')
        plt.ylabel('significance score')
        plt.savefig(os.path.join(save_fig_path, 'folder' + str(folder_no) + 'gf_general_plot'))
        plt.close()

        print('Complete save plots for folder_no:', folder_no)


def gen_list(feature_path, img_dir_base, classes, filter_size):
    """
    Function to generate the list of genes to submit to the DAVID platform
    :param feature_path: path of "feature_name.csv"
    :type feature_path: str
    :param img_dir_base: base path for output
    :type img_dir_base: str
    :param classes: list of classes names
    :type classes: list
    :authors: Nicola Pagliara; Gaetano Antonucci
    """
    GENE_FILTER_SIZE = filter_size  # maximum number of genes to submit
    num_genes = pd.DataFrame(pd.read_csv(feature_path, sep=',', header=None))
    list_complete_gene_list = []
    for i in range(1, int(str(num_genes[0].shape[0])) - 1):  # i=1 salto header di feature_name.csv
        complete_genes = num_genes.iloc[i][1]
        symbol, gene_id = str(complete_genes).split("|", 1)
        list_complete_gene_list.append(symbol)

    for folder_no in range(0, support.NUM_FOLDS):
        # Dictionary creation <key: pixel intensity in the heatmap of Ggcam img tumor class, value: pixel related>
        dictonaries = []

        for cls in range(0, len(classes)):
            dictonary_cls = OrderedDict()
            dataframe_cls_score = pd.DataFrame(
                pd.read_csv(img_dir_base + str(folder_no) + '/ConfidenceScore/' + classes[cls] + '_confidence.csv', sep=',',
                            header=None), dtype=int)

            cls_arry = []

            for i in range(0, int(str(num_genes[0].shape[0])) - 1):
                cls_arry.append(dataframe_cls_score.iloc[i][0])

            for j in range(0, np.size(cls_arry) - 1):
                dictonary_cls[list_complete_gene_list[j]] = cls_arry[j]

            dict_cls_sort = OrderedDict(sorted(dictonary_cls.items(), key=itemgetter(1), reverse=True))
            dictonaries.append(dict_cls_sort)

        # dictionary_rank_genes-confidence_score
        for num in range(0, len(dictonaries)):
            df = pd.DataFrame(dictonaries[num], index=[0])
            if not os.path.exists(img_dir_base + str(folder_no) + '/top_genes/'):
                os.makedirs(img_dir_base + str(folder_no) + '/top_genes/')
            df.to_csv(img_dir_base + str(folder_no) + '/top_genes/' + classes[
                num] + '_dictionary_rank_genes-confidence_score.csv', header=False, index=False)

        classes_filtered_lists = []
        for cls in range(0, len(classes)):
            filter_list = []
            counter_filter = 0
            for key in dictonaries[cls]:
                filter_list.append(key)
                counter_filter += 1
                if counter_filter == GENE_FILTER_SIZE:
                    break
            classes_filtered_lists.append(filter_list)

        # saving top 400 genes list in a .txt file for submission
        path_file = img_dir_base + str(folder_no)
        for cls in range(0, len(classes)):
            filename = path_file + '/top_genes/' + classes[cls] + '_top400thgenes_fold' + str(folder_no) + '.txt'
            if os.path.exists(filename):
                os.remove(filename)

            file = open(filename, 'w')
            for gene_name in classes_filtered_lists[cls]:
                file.write(gene_name + "\n")
            file.close()

        print("Complete save files for DAVID Platform for folder_no: ", folder_no)
    # List union to obtain one txt file to submit to DAVID
    lst_genes = []  # list of dictionaries
    for i in range(0, len(classes)):
        class_gene = set()
        for j in range(0, support.NUM_FOLDERS):
            path_file = img_dir_base + str(j)
            new_loc = os.path.join(path_file, 'top_genes')
            filename = os.path.join(new_loc, classes[i] + '_top400thgenes_fold' + str(j) + '.txt')
            file = open(filename, 'r')
            lines = file.readlines()
            for line in lines:
                class_gene.add(line)
        print('Set dimension of ', classes[i], ': ', len(class_gene))
        lst_genes.append(class_gene)

    path_file = ((img_dir_base.split('/img_fold'))[0]) + '/DAVID_files/top_genes'
    if not os.path.exists(path_file):
        os.makedirs(path_file)
    for z in range(0, len(lst_genes)):
        if len(lst_genes[z]) > 3000:
            print('CAUTION !!!, 3000 genes limit passed for class ' + classes[z])
        wr_file = open(os.path.join(path_file, classes[z] + '_genesToSubmit.txt'), 'w')
        wr_file.writelines(lst_genes[z])
        wr_file.close()
    print('LIST GENERATION END')


def binary_test_validation():
    # img_dir_base = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\GAETANO_Data_mapped_images\\binary_test\\img_fold'
    img_dir_base = 'C:\\Users\\gnoan\\Downloads\\Dati\\binary_test_03042023\\img_fold'
    classes = ['DLBC', 'UCS']
    # feature_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\GAETANO_Data_mapped_images\\binary_test\\feature_name.csv'
    feature_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\binary_test_03042023\\feature_name.csv'
    # gen confidence score csv
    gene_confidence_score_heatmaps(img_dir_base, classes)
    # plot generation
    gen_plot(feature_path=feature_path, img_dir_base=img_dir_base, classes=classes)
    # list generation
    filter_size = 400
    gen_list(feature_path=feature_path, img_dir_base=img_dir_base, classes=classes, filter_size=filter_size)


def ternary_test_validation():
    # img_dir_base = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\GAETANO_Data_mapped_images\\ternary_test\\img_fold'
    img_dir_base = 'C:\\Users\\gnoan\\Downloads\\Dati\\ternary_test_22032023\\img_fold'
    classes = ['BLCA', 'CESC', 'LGG']
    # feature_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\GAETANO_Data_mapped_images\\ternary_test\\feature_name.csv'
    feature_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\ternary_test_22032023\\feature_name.csv'
    # gen confidence score csv
    # gene_confidence_score_heatmaps(img_dir_base, classes)
    # plot generation
    # gen_plot(feature_path=feature_path, img_dir_base=img_dir_base, classes=classes)
    # list generation
    filter_size = 400
    gen_list(feature_path=feature_path, img_dir_base=img_dir_base, classes=classes, filter_size=filter_size)


def general_test_validation():
    img_dir_base = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/data_mapped_images/img_fold'
    # img_dir_base = 'C:\\Users\\gnoan\\Downloads\\Dati\\data_mapped_images\\img_fold'
    img_dir_base = 'C:\\Users\\gnoan\\Downloads\\Dati\\Dati_GPU_(ultima esecuzione)\\data_mapped_images\\img_fold'
    classes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP',
               'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
               'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    feature_path = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/data_mapped_images/feature_name.csv'
    # feature_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\data_mapped_images\\feature_name.csv'
    feature_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\Dati_GPU_(ultima esecuzione)\\data_mapped_images\\feature_name.csv'
    # gen confidence score csv
    gene_confidence_score_heatmaps(img_dir_base, classes)
    # plot generation
    gen_plot(feature_path=feature_path, img_dir_base=img_dir_base, classes=classes)
    # list generation
    filter_size = 400
    gen_list(feature_path=feature_path, img_dir_base=img_dir_base, classes=classes, filter_size=filter_size)


def main():
    # binary_test_validation()
    # ternary_test_validation()
    general_test_validation()


if __name__ == '__main__':
    main()
