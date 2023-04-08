"""
Support class for DL pipeline
"""
import numpy as np
import pandas as pd
from skimage import io

"""
CONSTANTS
"""
BINARY_CLASSES = ['DLBC', 'UCS']
TERNARY_CLASSES = ['BLCA', 'CESC', 'LGG']
CLASSES = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML','LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT','THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
NUM_FOLDS = 10

"""
Support for module Visualitaiton
"""

def zero_pic(img):
    if img.max() == 0 and img.min() == 0:
        return True
    else:
        return False


# the opencv image channel is BGR
def blue_pic(img):
    B_channel = img[:, :, 0]
    G_channel = img[:, :, 1]
    R_channel = img[:, :, 2]

    reverse_B = 255 - B_channel
    if zero_pic(reverse_B) and zero_pic(G_channel) and zero_pic(R_channel):
        return True
    else:
        return False

"""
Support for model Validation
"""

def transform_img_to_data(fold_idx, cls_name):
    """
    This function transform pixel intensity in confidence_score
    :param fold_idx:
    :param cls_name:
    :return:
    """
    confidence_score = []
    folder_path = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/data_mapped_images/img_fold'
    img_path = folder_path + str(fold_idx) + '/' + 'heatmaps' + '/AvgGuidedGradCam/' + cls_name + '_avg_guided_gcam_norm.png'
    im = io.imread(img_path)
    tmp = im - im.min()
    tmp_2 = tmp / tmp.max()
    tmp_3 = np.uint8(tmp_2 * 255.0)
    tmp_4 = tmp_3.flatten()
    confidence_score.append(tmp_4)
    D = pd.DataFrame(confidence_score).transpose()
    return D
