from torchvision import transforms

import support.classes as supcls
from support import support
from tqdm import tqdm

# training parameters
num_epochs = 200
batch_size = 90  # 175 #200 #500 these commented batch size has fail to train the model because they generate CUDA OutOfMemory error caused by memory fragmation
learning_rate = 0.0001

root_test_path = '/home/musimathicslab/PycharmProjects/pagliara_antonucci_progetto_sfb-gaetano/data_mapped_images/img_fold'

def generate_dataloader(fold_idx):
    """
    This function generate DataLoader for the test
    :param fold_idx: folder index
    :type fold_idx: int
    :return: dataloader_train
    """
    train_csv_path = root_test_path + str(fold_idx) + '/train/labels_train.csv'
    test_csv_path = root_test_path + str(fold_idx) + '/test/labels_test.csv'
    test_dataset = supcls.LocalTumorDatasetTest(csv_file=test_csv_path,
                                                root_dir=root_test_path + str(fold_idx) + '/test',
                                                transform=transforms.Compose([supcls.ToTensor()]))
    train_dataset = supcls.LocalTumorDatasetTrain(csv_file=train_csv_path,
                                                  root_dir=root_test_path + str(fold_idx) + '/train',
                                                  transform=transforms.Compose([supcls.ToTensor()]))
    dataloader_train = supcls.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = supcls.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader_train, dataloader_test

def model_train(num_class_model, fold_idx):
    """
    This function perform the training
    :param num_class_model: number of classes under test
    :type num_class_model: int
    :param fold_idx: folder index
    :type fold_idx: int
    :return: None
    """
    # model = supcls.Net(num_of_classes=num_class_model)
    model_var = supcls.VarNet(num_of_classes=num_class_model)
    model_var = model_var.double()
    model_var.cuda()
    running_loss = []
    # criterion = supcls.nn.CrossEntropyLoss()
    criterion_var = supcls.nn.MultiMarginLoss()  # seems useful for multi-class/multi-label classification for VarNet
    # optimizer = supcls.torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    optimizer_var = supcls.torch.optim.NAdam(params=model_var.parameters(),
                                             lr=learning_rate)  # Nadam combine Adam plus NAG for VarNet
    local_dataloader_train, local_dataloader_test = generate_dataloader(fold_idx)

    for epoch in tqdm(range(num_epochs)):
        running_loss_tmp = 0.0
        for i, sample in enumerate(local_dataloader_train):
            images = sample['image']
            images.requires_grad = True
            images = images.cuda()
            labels = sample['label']
            labels = labels.view(-1)
            # labels = supcls.f.one_hot(labels, num_classes=num_class_model)  # mappare in valori nei valori concessi da num_classes nel tenosore label esistono valori 7 e 32
            labels = labels.type(
                supcls.torch.FloatTensor)# converto il tensore codifica in float dato che abbiamo impostato questo tipo con net.double().
            labels.requires_grad = True
            labels = labels.type(supcls.torch.LongTensor)
            labels = labels.cuda()
            # optimizer.zero_grad()
            optimizer_var.zero_grad()
            # outputs = model(images)
            outputs = model_var(images)
            # loss = criterion(outputs, labels)
            loss = criterion_var(outputs, labels)
            loss.backward()
            # optimizer.step()
            optimizer_var.step()
            running_loss_tmp += loss.data
        print('epoch', epoch, ':loss is', running_loss_tmp)
        running_loss.append(running_loss_tmp)
        if (epoch > 3) and (abs(running_loss[epoch] - running_loss[epoch - 1]) <= 0.0001) and (
                abs(running_loss[epoch - 1] - running_loss[epoch - 2]) <= 0.0001):
            break
    print('Training finished, for folder_no: ', fold_idx)
    # supcls.torch.save(model.state_dict(), root_test_path + str(fold_idx) + '/network_weights.pth')
    supcls.torch.save(model_var.state_dict(), root_test_path + str(fold_idx) + '/network_variant_weights.pth')
    model_test(num_class_model, fold_idx, local_dataloader_test)


def model_test(num_class_model, fold_idx, dataloader_test):
    """
    This function perfom testing
    :param num_class_model: number of classes under test
    :type num_class_model: int
    :param fold_idx: folder index
    :type fold_idx: int
    :param dataloader_test: DataLoader of test dataset
    :type dataloader_test: torch.utils.data.DataLoader
    :return: None
    """
    # model = supcls.Net(num_of_classes=num_class_model)
    # model = model.double()
    model_var = supcls.VarNet(num_of_classes=num_class_model)
    model_var = model_var.double()
    model_var.cuda()
    # parte di testing
    # model.load_state_dict(supcls.torch.load(root_test_path + str(fold_idx) + '/network_weights.pth'))
    model_var.load_state_dict(supcls.torch.load(root_test_path + str(fold_idx) + '/network_variant_weights.pth'))
    correct = 0
    total = 0
    predicted_save = supcls.np.array([])
    test_label_save = supcls.np.array([])

    for ii, test_sample in enumerate(dataloader_test):
        test_imgs = test_sample['image']
        test_imgs = test_imgs.cuda()
        test_label = test_sample['label']
        # outputs = model(test_imgs)
        outputs = model_var(test_imgs)
        _, predicted = supcls.torch.max(outputs.data, 1)
        total += test_label.size(0)
        predicted_tmp = predicted.cpu().numpy()
        test_label_tmp = test_label.squeeze().data.cpu().numpy()
        correct += (predicted_tmp == test_label_tmp).sum()
        predicted_save = supcls.np.append(predicted_save, predicted_tmp)
        test_label_save = supcls.np.append(test_label_save, test_label_tmp)

    supcls.pd.DataFrame(predicted_save).to_csv(root_test_path + str(fold_idx) + '/test/predicted_label_run_th1.csv',
                                               header=False, index=False)
    supcls.pd.DataFrame(test_label_save).to_csv(root_test_path + str(fold_idx) + '/test/test_label_run_th1.csv',
                                                header=False, index=False)
    print('Accuracy of this fold is %d %%' % (100 * correct / total))
    print("Test finished for folder_no:  ", fold_idx)


def main():
    for i in range(0, support.NUM_FOLDS):
        model_train(33, i)


if __name__ == '__main__':
    main()
