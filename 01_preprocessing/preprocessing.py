import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from skimage import io
from sklearn.feature_selection import VarianceThreshold
from skimage import img_as_ubyte
from imblearn.over_sampling import SVMSMOTE

img_size = 102 # to set with reference to variance threshold

# AUTHORS FUNCTION
def feature_selection(big_table, annotation_path, preprocessed_file_path, threshold):
    """
    Feature selection from raw data after filtering with annotation file and sorting by chromosomic number
    :param big_table: raw data
    :type big_table: pandas.DataFrame
    :param annotation_path: path of the annotation file for gene filtering
    :type annotation_path: str
    :param preprocessed_file_path: outputh path
    :type preprocessed_file_path: str
    :param threshold: threshold for gene similarity
    :type threshold: float
    :return: features, labels
    """
    # trasformo i dati raw in una lista
    # con questa operazione viene acquisita solo la prima riga di big_table
    # feature_name corrisponde ai nomi dei geni
    # la seconda colonna di big_table è nella forma "<Symbol>|<GeneID>"
    feature_name = list(big_table)
    # prendo le colonne dalla 2^ colonna all'ultima
    # la prima colonna si esclude perché contiene le labels (le classi tumorali - valori a 1 a 33)
    feature_name = feature_name[1:]
    # iloc = Purely integer-location based indexing for selection by position.
    # acquisisce la prima colonna [indice 0] (colonna delle label = classi tumorali -> valori da 1 a 33)
    labels = np.array(big_table.iloc[:, 0])
    # leggo il file di annotation
    # annotation è un DataFrame
    annotation = pd.read_csv(annotation_path, dtype=str)
    # recupero tutti i GeneID dal file di annotation
    gene_id_annotation = list(annotation.loc[:, "GeneID"])
    # recupero tutti # di cromosoma che servono per l'ordinamento
    gene_id_chr_annotation = list(annotation.loc[:, "chromosome"])
    # La lista seguente conterrà i geneId originali dei raw data
    # (alcuni dei quali non sono presenti nel file di annotation)
    gene_id_original = []
    idx1 = []  # indice dei geni nel raw che trovano una corrispondenza nel file di annotation
    idx1_annotation = []  # indice del geneId del file di annotation dei geni che sono anche in raw data
    k = 0 # indice di progresso
    # nel for si verifica quali geni dei dati raw sono presenti anche nel file di annotation
    for name in feature_name:
        # il ",1" in split serve a limitare il numero di split
        # symbol viene estratta ma non serve in quanto non è mai usata
        symbol, gene_id = name.split("|", 1)
        #print("[GNO] Symbol: ", symbol, "gene id: ", gene_id)
        gene_id_original.append(gene_id)
        if gene_id in gene_id_annotation:
            idx1.append(k)
            # indice del gene nel file di annotation
            idx1_annotation.append(gene_id_annotation.index(gene_id))
        print('compare with annotation, progress : %.2f%%' % ((k / len(feature_name))*100))
        k = k + 1
    # features_raw prende tutte le righe e le colonne dalla seconda all'ultima (si escludono solo le labels)
    # dei dati raw (big_table)
    # il tipo è float in quanto i dati sono numeri reali (conteggio normalizzato)
    features_raw = np.array(big_table.iloc[:, 1:], dtype=float)
    # siccome il range dei valori è enorme è stata applicata una trasformazione ai dati per ridurre la scala
    features = np.log2(1.0 + features_raw)
    # I valori inferiori a 1 sono stati tutti posti a 0 (zero) in quanto sono ritenuti rumore
    features[np.where(features <= 1)] = 0
    # [AUTORI] values corresponding to existing genes (1st filtering)
    # tutti i geni che hanno trovato una corrispondenza nel file di annotation vengono
    # utilizzati come features (dopo la trasformazione con log2 e l'eliminazione del rumore)
    # features[:, idx1_tmp] seleziona solo i geni (le colonne) che sono presenti in idx1
    features_filtered = np.array(list(features[:, idx1_tmp] for idx1_tmp in idx1)).transpose()
    # recupero i nomi delle feature_filtrate
    feature_name_filtered = list(feature_name[i] for i in idx1)
    # recupero il gene_id delle feature filtrate
    gene_id_chr = list(gene_id_chr_annotation[i] for i in idx1_annotation)
    # sort the features based on the chr number
    idx_sorted = sort_feature(gene_id_chr)
    # ordino i nomi delle feature
    feature_name_sorted = list(feature_name_filtered[j] for j in idx_sorted)
    # features_filtered[:, i] for i in idx_sorted prende i geni in maniera ordinata
    features_sorted = np.array(list(features_filtered[:, i] for i in idx_sorted)).transpose()
    print('features have been sorted based on chromosome')
    # [GNO] features with a training-set variance lower than this threshold will be removed
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(features_sorted)
    # maschera di booleani
    idx2 = selector.get_support()
    # recupero gli indici delle features selezionate (array di interi)
    idx2_num = selector.get_support(indices=True)
    # [AUTORI] numpy is different from list
    features = features_sorted[:, idx2]
    feature_name_final = list(feature_name_sorted[i] for i in idx2_num)
    feature_name_path = os.path.join(preprocessed_file_path, 'feature_name.csv')
    pd.DataFrame(feature_name_final).to_csv(feature_name_path)
    print('features are selected, the selected gene name are saved at', feature_name_path)
    print("|features| = ", len(feature_name_final))
    return features, labels


def sort_feature(chr_filtered):
    """
    A function to sort the feature selected by chromosomic order
    :param chr_filtered:
    :type chr_filtered: list
    :return: index of feature ordered by chromosomic order
    """
    idx_list = list(range(len(chr_filtered)))
    big_list_chr = zip(idx_list, chr_filtered)  # combine a list of indices and a list of chromosomes
    tmp = sorted(big_list_chr, key=lambda x: x[1])  # sort based on the chromosome
    idx, _ = zip(*tmp)
    return idx


def kfold_split(big_table, annotation_path, preprocessed_file_path, folds, threshold):
    features, labels = feature_selection(big_table, annotation_path, preprocessed_file_path, threshold)
    print('features have been read')
    # Stratified K-Folds cross-validator.
    # Provides train/test indices to split data in train/test sets.
    # This cross-validation object is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class.
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    # training and testing set generation
    index_list = list(skf.split(features, labels))
    print('END extracting index for training and testing')
    return features, labels, index_list


def embedding_2d(features_train, features_test, fold_idx, preprocessed_file_path, labels_train, labels_test):
    """
    This function embed data in 2D images.
    :param features_train:
    :param features_test:
    :param fold_idx: fold index
    :type fold_idx: int
    :param preprocessed_file_path:
    :type preprocessed_file_path: str
    :param labels_train:
    :param labels_test:
    :return:
    """
    print('embedding to 2D image, fold:', fold_idx)
    features_padded = np.zeros(img_size*img_size)
    subfolder_path = os.path.join(preprocessed_file_path + '/img_fold' + str(fold_idx))
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    subfolder_train_path = subfolder_path + '/train'
    if not os.path.exists(subfolder_train_path):
        os.makedirs(subfolder_train_path)
    subfolder_test_path = subfolder_path + '/test'
    if not os.path.exists(subfolder_test_path):
        os.makedirs(subfolder_test_path)
    pd.DataFrame(labels_train).to_csv(os.path.join(subfolder_train_path, 'labels_train.csv'))
    pd.DataFrame(labels_test).to_csv(os.path.join(subfolder_test_path, 'labels_test.csv'))
    for i in range(features_train.shape[0]):
        features_padded[range(features_train.shape[1])] = features_train[i, :]/max(features_train[i, :])
        features_train_tmp = features_padded.reshape(img_size, img_size)
        file_path = subfolder_train_path + '/'+str(i) + '.png'
        # next line is useful for avoiding Lossy conversion warning
        features_train_tmp = img_as_ubyte(features_train_tmp)
        io.imsave(file_path, features_train_tmp)
    for j in range(features_test.shape[0]):
        features_padded[range(features_test.shape[1])] = features_test[j, :]/max(features_test[j, :])
        features_test_tmp = features_padded.reshape(img_size, img_size)
        file_path = subfolder_test_path + '/' + str(j) + '.png'
        # next line is useful for avoiding Lossy conversion warning
        features_test_tmp = img_as_ubyte(features_test_tmp)
        io.imsave(file_path, features_test_tmp)

def over_sampling(training_data, training_label):
    """
    This function is used only for classification
    :param training_data:
    :param training_label:
    :return: The training data and training label after oversampling using SMOTE algorithm
    """
    training_data_resampled, training_label_resampled = SVMSMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5, n_jobs=12)\
        .fit_resample(training_data, training_label)
    return training_data_resampled, training_label_resampled
# ================================================================================================================
# OUR FUNCTIONS

def run_preprocessing(big_table, annotation_path, preprocessed_file_path, threshold):
    """
    This function starts the preprocessing
    :param big_table: raw data under preprocessing
    :param annotation_path: path of the annotation file
    :param preprocessed_file_path: output path
    :return: None
    """
    folds = 10
    features, labels, index_list = kfold_split(big_table, annotation_path, preprocessed_file_path, folds, threshold)
        # use of index_list
        # first index = fold index (from 0 to 9)
        # second index = 0: training and 1: testing
    for fold_idx in range(folds):
        # extraction of training and testing sets
        fold_training_set_index = index_list[fold_idx][0] #type = np.ndarray
        fold_test_set_index = index_list[fold_idx][1]     #type = np.ndarray
        #features_train and features_test have 10381 columns (each column is a gene)
        features_train = np.ndarray(shape=(len(fold_training_set_index), features.shape[1]))
        features_test = np.ndarray(shape=(len(fold_test_set_index), features.shape[1]))
        # labels extraction
        # labels_train and labels_test have only one column
        labels_train = np.ndarray(shape=len(fold_training_set_index), dtype=int)
        labels_test = np.ndarray(shape=len(fold_test_set_index), dtype=int)
        for i in range(0, len(fold_training_set_index)):
            # features_train extraction
            for j in range(0, features.shape[1]):
                tmp = features[fold_training_set_index[i]]
                features_train[i][j] = tmp[j]  #type = np.ndarray
            # labels_train extraction
            labels_train[i] = labels[fold_training_set_index[i]]

        for i in range(0, len(fold_test_set_index)):
            # features_test extraction
            for j in range(0,features.shape[1]):
                tmp = features[fold_test_set_index[i]]
                features_test[i][j] = tmp[j]  #np.ndarray
            # labels_test extraction
            labels_test[i] = labels[fold_test_set_index[i]]

        feature_train_resampled, label_train_resampled = over_sampling(features_train, labels_train)
        # images generation for convolutional neural network
        embedding_2d(feature_train_resampled, features_test, fold_idx, preprocessed_file_path, label_train_resampled, labels_test)
    print("END IMAGES GENERATION")


def binary_test_preprocessing():
    """
    This function set up the preprocessing of binary test
    :return:
    """
    # dataset's path
    big_table_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\raw_data\\binary_test_raw_data.csv'
    big_table = pd.read_csv(big_table_path)
    # annotation file's path
    annotation_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\raw_data\\04072018_annotation.csv'
    # output path
    preprocessed_file_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\GAETANO_Data_mapped_images\\binary_test'
    # variance threshold used in feature_selection
    threshold = 0.9552
    run_preprocessing(big_table=big_table, annotation_path=annotation_path, preprocessed_file_path=preprocessed_file_path, threshold=threshold)
    print('END Preprocessing for binary test')


def ternary_test_preprocessing():
    """
    This function set up the preprocessing of ternary test
    :return:
    """
    # dataset's path
    big_table_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\Classi\\classe_2_4_15.csv'
    big_table = pd.read_csv(big_table_path)
    # annotation file's path
    annotation_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\RawData\\RawDatafromFirehose\\04072018_annotation.csv'
    # output path
    preprocessed_file_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\GAETANO_Data_mapped_images\\ternary_test'
    # variance threshold used in feature_selection
    threshold =0.9869
    run_preprocessing(big_table=big_table, annotation_path=annotation_path,
                      preprocessed_file_path=preprocessed_file_path, threshold=threshold)
    print('END Preprocessing for ternary test')


def general_test_preprocessing():
    """
       This function set up the preprocessing of general test (33 tumor classes)
       :return:
       """
    # dataset's path
    big_table_path = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/raw_data/dataset.csv'
    big_table = pd.read_csv(big_table_path)
    # annotation file's path
    annotation_path = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/raw_data/04072018_annotation.csv'
    # output path
    preprocessed_file_path = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/data_mapped_images'
    # variance threshold used in feature_selection
    threshold = 1.19 # this threshold is the same used by Lyu and Haque
    run_preprocessing(big_table=big_table, annotation_path=annotation_path,
                      preprocessed_file_path=preprocessed_file_path, threshold=threshold)
    print('END Preprocessing for general test')

def main():
    #binary_test_preprocessing()
    ternary_test_preprocessing()
    #general_test_preprocessing()

if __name__ == '__main__':
    main()