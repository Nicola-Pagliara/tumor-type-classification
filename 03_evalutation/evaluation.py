import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

from support import support

def gather_test_result(root_path):
    """
    Function to gather test's result
    :param root_path: test root path
    :type root_path: str
    :return: None
    """
    predicted_label = []
    true_label = []
    for i in range(0, support.NUM_FOLDERS):
        predicted_label_path = root_path + str(i) + '/test/predicted_label_run_th1.csv'
        true_label_path = root_path + str(i) + '/test/test_label_run_th1.csv'
        predicted_label_tmp = np.array(pd.read_csv(predicted_label_path, header=None)) + 1.0
        true_label_tmp = np.array(pd.read_csv(true_label_path, header=None)) + 1.0
        predicted_label.append(predicted_label_tmp)
        true_label.append(true_label_tmp)
        true_label_all = np.concatenate(true_label, axis=0)
        predicted_label_all = np.concatenate(predicted_label, axis=0)
        print('RESULTS GATHERED')
        return true_label_all, predicted_label_all


def draw_confusion_matrix(predicted_label_all, true_label_all, classes, root_test_path):
    """
    Function to draw the confusion matrix
    :param predicted_label_all: predicted labels
    :param true_label_all: true labels
    :param classes: classes under test
    :type classes: list
    :param root_test_path: test root path
    :type root_test_path: str
    :return: cnf_matrix
    """
    cnf_matrix = confusion_matrix(true_label_all, predicted_label_all)
    cnf_matrix = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix of ' + str(len(classes)) + ' tumor types')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not os.path.exists(os.path.join(root_test_path, 'Cnf_Matrix')):
        os.makedirs(os.path.join(root_test_path, 'Cnf_Matrix'))
    plt.savefig(os.path.join(root_test_path, 'Cnf_Matrix', 'cnf_matrix.png'))
    print('CONFUSION MATRIX SAVED')
    return cnf_matrix


def compute_eval_metrics(true_label_all, predicted_label_all, classes, root_test_path):
    """
    Function to compute evaluation metrics
    :param true_label_all: true labels
    :param predicted_label_all: predicted labels
    :param classes: classes under test
    :type classes: list
    :param root_test_path: test root path
    :type root_test_path: str
    :return:
    """
    cnf_matrix = draw_confusion_matrix(predicted_label_all, true_label_all, classes, root_test_path)
    precison_s = precision_score(true_label_all, predicted_label_all, average='weighted')
    recall_s = recall_score(true_label_all, predicted_label_all, average='weighted')
    f1 = f1_score(true_label_all, predicted_label_all, average='weighted')
    accuracy = accuracy_score(true_label_all, predicted_label_all)
    accuracy_each_class = cnf_matrix.diagonal()
    metric_eval = [accuracy, precison_s, recall_s, f1]
    for i in range(0, len(accuracy_each_class)):
        metric_eval.append(accuracy_each_class[i])
    for i in range(0, cnf_matrix.shape[1]):
        metric_eval.append(cnf_matrix[i])
    print('EVALUATION METRICS COMPUTED')
    return metric_eval


def save_metrics_eval(eval_metrics, classes, header_label, root_test_path):
    """
    Function to save evaluation metrics
    :param eval_metrics: evaluation metrics
    :param classes: classes under test
    :type classes: list
    :param header_label: header
    :type header_label: list
    :param root_test_path: test root path
    :type root_test_path: str
    :return: None
    """
    index_label = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    eval_save_file = root_test_path + '/evaluation_' + str(len(classes)) + '_DA_class.csv'
    for i in range(0, len(classes)):
        index_label.append('Accuracy Cohort:' + classes[i])
    for i in range(0, len(classes)):
        index_label.append('Cnf-Matrix-Row ' + str(i + 1))
    csv_metric_eval = pd.DataFrame(eval_metrics, index=index_label)
    csv_metric_eval.to_csv(eval_save_file, sep='=', header=header_label)
    print('EVALUATION METRICS SAVED')

def main():
    # root_path = '/home/nicola/Unisa Magistrale/BioInformatica/Progetto/Code Paper Riferimento/DL-based-Tumor-Classification-master/model/Local Preprocessed Image/Ternary Class Preprocessed image/img_fold'
    # root_test_path = '/home/nicola/Unisa Magistrale/BioInformatica/Progetto/Code Paper Riferimento/DL-based-Tumor-Classification-master/model/Local Preprocessed Image/Ternary Class Preprocessed image'
    # root_test_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\GAETANO_Data_mapped_images\\ternary_test'
    # root_path = 'C:\\Users\\gnoan\\Documents\\ProgettoSFB\\GAETANO_Data_mapped_images\\ternary_test\\img_fold'
    # root_test_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\binary_test_03042023'
    # root_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\binary_test_03042023\\img_fold'
    # root_test_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\ternary_test_22032023'
    # root_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\ternary_test_22032023\\img_fold'
    root_test_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\data_mapped_images'
    root_path = 'C:\\Users\\gnoan\\Downloads\\Dati\\data_mapped_images\\img_fold'
    # classes = ['BLCA', 'CESC', 'LGG']
    # classes = ['DLBC', 'UCS']
    classes = support.CLASSES
    # header_label = ['10 Folds Model Evaluation on Ternary UnBalanced Case Classes: BLCA, CESC, LGG']
    # header_label = ['10 Folds Model Evaluation on Binary UnBalanced Case Classes: DLBC, UCS']
    header_label = ['10 Folds Model Evaluation on General Case']
    true, pred = gather_test_result(root_path)
    metric_evals = compute_eval_metrics(true, pred, classes, root_test_path)
    save_metrics_eval(metric_evals, classes, header_label, root_test_path)
    print('END of EVALUATION')

if __name__ == '__main__':
    main()
