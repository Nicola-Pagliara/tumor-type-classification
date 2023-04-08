import pandas as pd
import os
from itertools import repeat

# global path for retrive raw data
raw_data_root_path = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/raw_data'

def gather_data_txt(cls_name):
    """
    This function gather data from Illumina txt files
    :param cls_name: class name for gathering raw data of the class
    :return: pandas.DataFrame of raw data of the class
    """
    subpath = cls_name + '.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt'
    path_data = os.path.join(raw_data_root_path, cls_name, subpath)
    raw_datas = pd.DataFrame(pd.read_csv(path_data, delimiter='\t', header=None))
    return raw_datas

def format_raw_data(dataframe_data, indexs_drop, cls_name, ref_clss):
    """
    Formats the raw data into the correct form for testing
    :param dataframe_data: DataFrame to format
    :type dataframe_data: pandas.DataFrame
    :param indexs_drop: list of row to delete in dataframe_data
    :param cls_name: class name
    :param ref_clss: list of classes for the test
    :return: formatted pandas.DataFrame for testing
    """
    format_data = dataframe_data.drop(indexs_drop)
    format_data = format_data.transpose()
    shapes = format_data.shape[0] # numero di righe
    lst = list(repeat(ref_clss.index(cls_name) + 1, shapes)) #lista con 'numero di righe' elementi tutti uguali
    lst[0] = ""
    format_data.insert(0, '', lst, allow_duplicates=True) # aggiunto la colonna che rappresenta la classe tumorale
    save_path = os.path.join(raw_data_root_path, cls_name, cls_name + '.csv')
    format_data.to_csv(save_path, header=None, index=False)
    return format_data

def data_cleaning(df, cls_name):
    """
    Data cleaning to take only mRNASeq from TP (Primary Solid Tumor)
    :param df: pandas.DataFrame to clean
    :param cls_name: class name of tumor type
    :return:
    """
    list_samples = df.loc[0, :].values.flatten().tolist()
    print('list_samples: ', list_samples)
    subpath = os.path.join(raw_data_root_path,'REF_file_filter', cls_name + '_hybridization_ref_filter_not_spt.txt')
    df_clean = pd.read_csv(subpath, delimiter='\r\n', header=None)
    list_clean = df_clean[0].tolist()
    index=[]
    for ref_name in list_samples:
        for cl_name in list_clean:
            if cl_name == ref_name:
                index.append(list_samples.index(ref_name))
    print('clm index to delete: ', index)
    df = df.drop(index, axis=1)
    return df


# Unione di due dataframe mantendo il formato dato da format_raw_data
def data_union(df1, df2):
    df2.drop([0], inplace=True)
    df3 = df1.append(df2)
    return df3

def save_data(df_list, filename):
    df_to_save = pd.concat(df_list)
    df_to_save.info()
    df_to_save.to_csv(os.path.join(raw_data_root_path, filename), header=False, index=False)
def binary_test_data():
    binary_test_classes = ['DLBC', 'UCS']
    # composizione dati raw per test binario su DLBC e UCS
    # non è necessario fare il cleaning perché non ci sono valori da eliminare
    binary_raw_data = []
    for i in range(0, len(binary_test_classes)):
        print(i)
        df = gather_data_txt(cls_name=binary_test_classes[i])
        df = format_raw_data(dataframe_data=df, indexs_drop=[0, 1], cls_name=binary_test_classes[i],
                             ref_clss=binary_test_classes)
        binary_raw_data.append(df)

    # elimino la header row dal secondo dataframe
    binary_raw_data[1].drop([0], inplace=True)
    # salvo i dati
    save_data(df_list=binary_raw_data, filename='binary_test_raw_data.csv')

def ternary_test_data():
    ternary_test_classes = ['BLCA', 'CESC', 'LGG']
    # composizione dati raw per test ternario su BLCA, CESC e LGG
    # è necessario eseguire il cleaning dei dati in quanto sono presenti sample non appartenenti a TP
    ternary_raw_data = []
    for i in range(0, len(ternary_test_classes)):
        df = gather_data_txt(cls_name=ternary_test_classes[i])
        df = data_cleaning(df=df, cls_name=ternary_test_classes[i])
        df = format_raw_data(dataframe_data=df, indexs_drop=[0,1], cls_name=ternary_test_classes[i], ref_clss=ternary_test_classes)
        ternary_raw_data.append(df)

    # elimino la header row dal secondo e dal terzo dataframe
    ternary_raw_data[1].drop([0], inplace=True)
    ternary_raw_data[2].drop([0], inplace=True)
    # salvo i dati
    save_data(df_list=ternary_raw_data, filename='ternary_test_raw_data.csv')

def general_test_remapping():
    big_table = pd.read_csv(raw_data_root_path + '/big_data_2.csv')
    big_table.info()

    for i in range(0, 33):
        big_table.loc[big_table['Unnamed: 0'] == (i+1), 'Unnamed: 0'] = i

    big_table.to_csv(raw_data_root_path + '/dataset.csv', header=True, index=False)
    # CAUTION! csv file header changed by hand

def binary_test_remapping():
    big_table = pd.read_csv(raw_data_root_path + '/binary_test_raw_data.csv')
    big_table.info()

    for i in range(0, 2):
        big_table.loc[big_table['Unnamed: 0'] == (i + 1), 'Unnamed: 0'] = i

    big_table.to_csv(raw_data_root_path + '/binary_test_raw_data_remapped.csv', header=True, index=False)
    # CAUTION! csv file header changed by hand

def ternary_test_remapping():
    big_table = pd.read_csv(raw_data_root_path + '/ternary_test_raw_data.csv')
    big_table.info()

    for i in range(0, 3):
        big_table.loc[big_table['Unnamed: 0'] == (i + 1), 'Unnamed: 0'] = i

    big_table.to_csv(raw_data_root_path + '/ternary_test_raw_data_remapped.csv', header=True, index=False)
    # CAUTION! csv file header changed by hand
def main():
    # binary_test_data()
    # binary_test_remapping()
    # ternary_test_data()
    ternary_test_remapping()
    # general_test_remapping()

if __name__ == '__main__':
    main()

