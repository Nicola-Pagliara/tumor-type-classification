import glob
import os.path

import cv2
import numpy as np
from torchvision import transforms

from support import classes as supcls
from support import support

batch_size = 1
# path to retrieve data for Visualization with heatmaps
root_path = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/data_mapped_images/img_fold'
topk = 1
# The target_layer is 'conv3' for Net and 'conv4' for VarNet
# target_layer = 'conv3'
target_layer = 'conv4'
classes = support.CLASSES

def init_dataset_heatmaps(fold_idx):
    """
    Function to initialize dataset
    :param fold_idx: fold index
    :type fold_idx: int
    :return:
    """
    train_root_path = root_path + str(fold_idx) + '/train'
    train_csv_path = train_root_path + '/labels_train.csv'
    train_dataset = supcls.LocalTumorDatasetTrain(csv_file=train_csv_path, root_dir=train_root_path,
                                                   transform=transforms.Compose([supcls.ToTensor()]))
    loader_train = supcls.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader_train


def init_component_heatmaps(num_cls, fold_idx):
    """
    Function to set up the component need to generate heatmaps
    :param num_cls: number of classes under test
    :type num_cls: int
    :param fold_idx: fold index
    :type fold_idx: int
    :return:
    """
    state_path = os.path.join(root_path + str(fold_idx), 'network_variant_weights.pth')
    # CAUTION ! Make sure to choice the correct Model
    #model = supcls.Net(num_of_classes=num_cls)
    model = supcls.VarNet(num_of_classes=num_cls)
    model = model.double()
    model.load_state_dict(supcls.torch.load(state_path))
    model.eval()
    gcam = supcls.GradCAM(model=model)
    gbp = supcls.GuidedBackPropagation(model=model)
    return gcam, gbp


def generate_heatmaps(gcam, gbp, train_loader, folder_no):
    """
    Main function to generate heatmaps
    :param gcam: GradCAM instance
    :type gcam: support.GradCAM
    :param gbp: GuidedBackPropagation instance
    :type gbp: support.GuidedBackPropagation
    :param train_loader: data loader
    :type train_loader: torch.utils.data.DataLoader
    :param folder_no: fold index
    :type folder_no: int
    :return:
    """
    count = 0
    for batch_idx, diction in enumerate(train_loader):
        images = diction['image']
        images.requires_grad = True
        images = images.double()
        probs, ids = gcam.forward(images)
        _ = gbp.forward(images)
        for j in range(topk):
            gcam.backward(idx=ids[j])
            regions = gcam.generate(target_layer=target_layer)
            gbp.backward(idx=ids[j])
            gradients = gbp.generate()
        count += 1
        print('Generate Gradient and Gcam Regions for batch_no: ', batch_idx)
        save_GradCam(regions, gcam, folder_no, ids, count)
        save_Guided_Gcam(regions, gradients, gbp, folder_no, ids, count)


def save_GradCam(regions, gcam, folder_no, ids, count):
    """
    Function to save GradCAM generated heatmaps.
    :param regions:
    :param gcam:
    :param folder_no:
    :param ids:
    :param count:
    :return:
    """
    for j in range(topk):
        gcam_directory = os.path.join(root_path + str(folder_no), "heatmaps_variant", "GradCam", classes[ids[j]])
        if not os.path.exists(gcam_directory):
            os.makedirs(gcam_directory)
        gcam_filename = os.path.join(gcam_directory, "{}-gradcam-{}.png".format(count, classes[ids[j]]))
        gcam.save(filename=gcam_filename, gcam=regions)
    print('Complete GradCam img for folder_no: ', folder_no)


def save_Guided_Gcam(regions, gradients, gbp, folder_no, ids, count):
    """
    Function to save GuidedGradCAM generated heatmaps.
    :param regions:
    :param gradients:
    :param gbp: GuidedBackpropagation instance
    :param folder_no: fold index
    :type folder_no: int
    :param ids:
    :param count: image number
    :type count: int
    :return:
    """
    for j in range(topk):
        gbp_directory = os.path.join(root_path + str(folder_no), "heatmaps_variant", "GuidedGradCam", classes[ids[j]])
        if not os.path.exists(gbp_directory):
            os.makedirs(gbp_directory)
        gbp_filename = os.path.join(gbp_directory, "{}-guided_gcam-{}.png".format(count, classes[ids[j]]))
        gbp.save(
            filename=gbp_filename,
            data=regions * gradients
        )
    print('Complete Guided GradCam for folder_no: ', folder_no)


def filter_null_img_Guided_GradCam():
    """
    Function to remove null images in GuidedGradCAM.
    :return:
    """
    for r in range(10):
        img_dir = root_path + str(r) + '/heatmaps_variant/GuidedGradCam'
        counter = 0
        for cs in classes:
            file_paths = glob.glob(img_dir + '/' + cs + '/*guided*')
            for file_path in file_paths:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if support.zero_pic(img):
                    os.remove(file_path)
                    counter += 1
        print('Image Guided gradCam eliminated from folder:', r)
        print('Number img delete is:', counter)


def filter_null_img_Gradcam():
    """
    Function to remove null images in GradCAM.
    :return:
    """
    for w in range(10):
        img_dir = root_path + str(w) + '/heatmaps_variant/GradCam'
        counter = 0
        for cs in classes:
            file_paths = glob.glob(img_dir + '/' + cs + '/*gradcam*')
            for file_path in file_paths:
                img = cv2.imread(file_path)
                if support.blue_pic(img):
                    os.remove(file_path)
                    counter += 1
        print('Image gradCam eliminated from folder:', w)
        print('Number img delete is:', counter)


def gen_avg_norm_GradCam():
    """
    Function to generate GradCAM average images for each class under test
    :return:
    """
    for k in range(0, 10):
        folder_no = k
        img_dir = root_path + str(folder_no) + '/heatmaps_variant/GradCam'
        output_img = root_path + str(folder_no) + '/heatmaps_variant/AvgGradCam/'
        for cs in classes:
            gcam_img = []
            file_paths = glob.glob(img_dir + '/' + cs + '/*gradcam*')
            for file_path in file_paths:
                img = cv2.imread(file_path)
                gcam_img.append(img)
            gcam_img = np.array(gcam_img)
            avg_gcam = gcam_img.mean(axis=0)
            avg_gcam -= avg_gcam.min()
            if avg_gcam.max() != 0:
                avg_gcam = avg_gcam / avg_gcam.max()
            avg_gcam_norm = avg_gcam * 255.0
            if not os.path.exists(output_img + cs):
                os.makedirs(output_img + cs)
            cv2.imwrite(output_img + cs + '_avg_gcam_norm_grad.png', np.uint8(avg_gcam_norm))
            print('Complete GradCam norm class for Coohrt:', cs)


def gen_avg_norm_GuidedGcam():
    """
    Function to generate GuidedGradCAM average images for each class under test.
    :return:
    """
    for s in range(0, 10):
        folder_no = s
        img_dirc = root_path + str(folder_no) + '/heatmaps_variant/GuidedGradCam'
        output_imgc = root_path + str(folder_no) + '/heatmaps_variant/AvgGuidedGradCam/'
        for cs in classes:
            guided_gcam = []
            file_paths = glob.glob(img_dirc + '/' + cs + '/*guided*')
            for file_path in file_paths:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                guided_gcam.append(img)

            guided_gcam = np.array(guided_gcam)
            avg_guided_gcam = guided_gcam.mean(axis=0)
            avg_guided_gcam = np.uint8(avg_guided_gcam)
            avg_guided_gcam -= avg_guided_gcam.min()

            if avg_guided_gcam.max() != 0:
                avg_guided_gcam = avg_guided_gcam / avg_guided_gcam.max()
            avg_guided_gcam_norm = avg_guided_gcam * 255.0

            if not os.path.exists(output_imgc + cs):
                os.makedirs(output_imgc + cs)
            cv2.imwrite(output_imgc + cs + '_avg_guided_gcam_norm.png', np.uint8(avg_guided_gcam_norm))
            print('Complete  Guided GradCam norm class for Coorht:', cs)


def main():
    # CAUTION ! Before run check the path in every function
    for i in range(0, support.NUM_FOLDS):
        loader_train = init_dataset_heatmaps(i)
        # CAUTION ! Make sure that num_cls in the number of tumor classes under test
        gcam, gbp = init_component_heatmaps(num_cls=33, fold_idx=i)
        generate_heatmaps(gcam=gcam, gbp=gbp, train_loader=loader_train, folder_no=i)

    filter_null_img_Gradcam()
    filter_null_img_Guided_GradCam()

    gen_avg_norm_GradCam()
    gen_avg_norm_GuidedGcam()


if __name__ == '__main__':
    main()
