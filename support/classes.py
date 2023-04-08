import os
from collections import OrderedDict


import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from skimage import img_as_float, io
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Test Classe TumorDataset Train,Test adeguata al caso generale

"""
 Modulo Model Train-Test 
"""


class LocalTumorDatasetTrain(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = np.array(pd.read_csv(csv_file, skiprows=1, sep=',', header=None)).astype(int)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = str(idx) + '.png'
        img_path = os.path.join(self.root_dir, img_name)
        img = np.empty(shape=(1, 102, 102))
        img[0, :, :] = (img_as_float(
            io.imread(img_path)) - 0.5) / 0.5          # riduce il numero di cifre a 4 per ogni valore del tensore del immagine
        label = np.array([self.labels_frame[idx, 1]])  # estrae il valore tensoriale della label
        train_sample = {'image': img, 'label': label}
        if self.transform:
            train_sample = self.transform(train_sample)
        return train_sample


"""
 Modulo Model Train-Test 
"""


class LocalTumorDatasetTest(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = np.array(pd.read_csv(csv_file, skiprows=1, sep=',', header=None)).astype(int)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = str(idx) + '.png'
        img_path = os.path.join(self.root_dir, img_name)

        "!!! Setting dimensione immagine !!!"
        img = np.empty(shape=(1, 102, 102))
        img[0, :, :] = (img_as_float(io.imread(img_path)) - 0.5) / 0.5
        label = np.array([self.labels_frame[
                              idx, 1]])  # viene tolto il -1 poichè questo e caso binario e comparirebbe la classe -1 che non viene trattata dal modello causando un abbassmento del accuracy
        test_sample = {'image': img, 'label': label}

        if self.transform:
            test_sample = self.transform(test_sample)
        return test_sample


"""
 Modulo Model Train-Test 
"""


class ToTensor(object):

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        return {'image': torch.from_numpy(image), 'label': torch.LongTensor(
            labels)}  # torch.LongTensor rapprensenta il cambio di tipo da tensore 1x102x102 in un intero a 64 bit, variante utilizata per le GPU


"""
 Modulo Model Train-Test 
"""


class Net(nn.Module):
    def __init__(self, num_of_classes):
        super(Net, self).__init__()
        # input image channel, output channels, kernel size square convolution
        # kernel
        # input size = 102, output size = 100
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # input size = 50, output size = 48
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # input size = 24, output size = 24
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.vp = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_of_classes)

    def forward(self, x):
        in_size = x.size(0)
        x = f.relu(self.bn1(self.vp(self.conv1(x))))
        x = f.relu(self.bn2(self.vp(self.conv2(x))))
        x = f.relu(self.bn3(self.vp(self.conv3(x))))
        x = self.drop2D(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    """
    Classe VarNet for enchanting accuracy of model
    """


class VarNet(nn.Module):
    def __init__(self, num_of_classes):
        super(VarNet, self).__init__()
        # input image channel, output channels, kernel size square convolution
        # kernel
        # input size = 102, output size = 100
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # input size = 50, output size = 48
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # input size = 24, output size = 24
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # input size = 12,  output size = 12
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.vp = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_of_classes)

    def forward(self, x):
        in_size = x.size(0)
        x = f.selu(self.bn1(self.vp(self.conv1(x))))
        x = f.selu(self.bn2(self.vp(self.conv2(x))))
        x = f.selu(self.bn3(self.vp(self.conv3(x))))
        x = f.selu(self.bn4(self.vp(self.conv4(x))))  # Selu must be explore because overcame the problem of negative slopes that afflict relu also the selu make the model converge faster than relu
        x = self.drop2D(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    """
    Classi del modulo Visualization

    """


class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.DoubleTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        self.probs = f.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.data.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_full_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.DoubleTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            res = fmap * weight.data.expand_as(fmap)
            gcam += fmap * weight.data.expand_as(fmap)
            gcam.requires_grad = False
            gcam = f.relu(gcam)

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        if gcam.max() != 0:
            gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        if gcam.max() != 0:
            gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


class BackPropagation(PropagationBase):
    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def generate(self):
        output = self.image.grad.data[0].numpy()[0]
        return output

    def save(self, filename, data):
        abs_max = np.maximum(-1 * data.min(), data.max())
        data = data / abs_max * 127.0 + 127.0
        cv2.imwrite(filename, np.uint8(data))


class GuidedBackPropagation(BackPropagation):

    def _set_hook_func(self):

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_in[0]

            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return torch.clamp(grad_in[0], min=0.0)

        for module in self.model.named_modules():
            module[1].register_full_backward_hook(func_b)  # PyTorch warning about using a non-full backward hook when the forward contains multiple autograd Nodes. Solved by changing the deprecated function
