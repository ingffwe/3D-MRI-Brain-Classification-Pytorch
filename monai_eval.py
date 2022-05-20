import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import CSVSaver, ImageDataset
from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, EnsureType

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, EnsureType
import matplotlib.pyplot as plt


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

classes = 2   # 类数
data_path = r'D:\3DCLFNii\txt'
train_txt_path = data_path + "\\train_fALFF.txt"
val_txt_path = data_path + "\\test_fALFF.txt"
seed = 1

def Txt_to_list_noshuffle(file_path):
    '''
    :param file_path: 单个网络包含图片地址和类标的txt文件
    :return: 单个网络的训练集和测试集
    '''
    # 1.读取图片地址
    with open(file_path, 'r') as f1:
        lines1 = f1.readlines()

    # 除去label
    img_list = []
    for img in lines1:
        img_list.append(img.split(' ')[0])

    # 2.读取label
    label_list = []
    for label in lines1:
        label_list.append(int(label.split(' ')[1].strip('\n')))

    return img_list, label_list
def Split_Data(txt_path_train,txt_path_val, seed=seed):
    # 1.这部分是按照个体进行打乱
    train_txt_path = txt_path_train
    val_txt_path = txt_path_val

    dir_list_noshuffle, label_list_noshuffle = Txt_to_list_noshuffle(train_txt_path)  #  根据Txt生成列表

    dir_list_noshuffle_array = np.array(dir_list_noshuffle)  # 转换为Numpy矩阵用于reshape
    label_list_noshuffle_array = np.array(label_list_noshuffle)

    # dir_list_noshuffle_array = np.reshape(dir_list_noshuffle_array, (116, ))  # 按个体reshape
    # label_list_noshuffle_array = np.reshape(label_list_noshuffle_array, (116, 1440))

    np.random.seed(seed)  # 按照个体随机打乱
    np.random.shuffle(dir_list_noshuffle_array)
    np.random.seed(seed)
    np.random.shuffle(label_list_noshuffle_array)
    X_train = dir_list_noshuffle_array
    Y_train = label_list_noshuffle_array

    # dir_list_shuffle_array = dir_list_noshuffle_array.flatten()  # 按照个体展开
    # label_list_shuffle_array = label_list_noshuffle_array.flatten()
    #
    # dir_list_shuffle = dir_list_shuffle_array.tolist()
    # label_list_shuffle = label_list_shuffle_array.tolist()

    # # 2.这部分是按照个体换划分数据集,之后混合所有个体的图片
    # split = 167040
    # X_train = dir_list_shuffle[:split]
    # np.random.seed(seed)
    # np.random.shuffle(X_train)
    # Y_train = label_list_shuffle[:split]
    # np.random.seed(seed)
    # np.random.shuffle(Y_train)

    #验证集可打乱
    dir_list_val, label_list_val = Txt_to_list_noshuffle(val_txt_path)
    X_valid = dir_list_val
    np.random.seed(seed)
    np.random.shuffle(X_valid)
    Y_valid = label_list_val
    np.random.seed(seed)
    np.random.shuffle(Y_valid)

    return X_train, Y_train, X_valid, Y_valid

train_data, train_label, valid_data, valid_label = Split_Data(train_txt_path, val_txt_path, seed=seed)
print(train_data[:10], train_label[:10])


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = 0
    max = 3000
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""

    #图像剪裁
    img = img[10:55,10:60,10:55]    #61,73,61
    # Set the desired depth
    desired_depth = 64
    desired_width = 64
    desired_height = 64
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def get_nib(paths):
    '''
        :param paths: 要读取的图片路径列表
        :return: 图片数组
        '''
    imgs = []

    for path in paths:
        img = read_nifti_file(path)  # 以三通道方式读取
        img = resize_volume(img)
        # img = normalize(img)
        img = img.astype('float32')


        imgs.append(img)

    return np.array(imgs).reshape(len(paths), 64, 64, 64, 1)

class CustomImageDataset(Dataset):
    def __init__(self, data_list, label_list, transform=None, target_transform=None):
        self.img_labels = label_list
        self.img_datas = data_list
        self.transform = transform
        self.target_transform = target_transform

    def get_nib(self, path):
        '''
            :param paths: 要读取的图片路径列表
            :return: 图片数组
            '''
        imgs = []


        img = read_nifti_file(path)  # 以三通道方式读取
        img = resize_volume(img)
        # img = normalize(img)
        img = img.astype('float32')

        imgs.append(img)

        return np.array(imgs).reshape(1, 64, 64, 64)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_datas[idx]
        image = self.get_nib(image)
        image = torch.tensor(image)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def main():
    val_loader = DataLoader(CustomImageDataset(valid_data, valid_label), batch_size=4, shuffle=True)

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)

    model.load_state_dict(torch.load("best_metric_model_classification3d_array.pth"))
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        # saver = CSVSaver(output_dir="./output")
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images).argmax(dim=1)
            value = torch.eq(val_outputs, val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            # saver.save_batch(val_outputs, val_data[1])
        metric = num_correct / metric_count
        print("evaluation metric:", metric)
        # saver.finalize()


if __name__ == "__main__":
    main()