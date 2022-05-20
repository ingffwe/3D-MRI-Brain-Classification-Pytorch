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
import pytorch_lightning as pl
from datetime import datetime
import torchio as tio
import time

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

classes = 2   # 类数
data_path = r'D:\3DCLFNii\txt'
train_txt_path = data_path + "\\train_fALFF.txt"
val_txt_path = data_path + "\\val_fALFF.txt"
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

# train_loader = DataLoader(CustomImageDataset(train_data, train_label), batch_size=4, shuffle=True)
# val_loader = DataLoader(CustomImageDataset(valid_data, valid_label), batch_size=4, shuffle=False)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # 在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # 实现数据集的定义，每张GPU都会执行该函数, stage 用于标记是用于什么阶段
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomImageDataset(train_data, train_label, transform=None)
            self.val_dataset = CustomImageDataset(valid_data, valid_label, transform=None)
        if stage == 'test' or stage is None:
            # self.test_dataset = CustomImageDataset(self.test_file_path, self.test_file_num, transform=None)
            pass
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
# Unet
unet = monai.networks.nets.unet(spatial_dims=3, in_channels=1, out_channels=2, channels=(8, 16, 32), strides=(2, 3)).to(device)

# ViT
# model = monai.networks.nets.ViT(in_channels=1, img_size=(64,64,64), patch_size=(16,16,16), pos_embed='conv')
# model = model.cuda()


class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.LABEL]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss


model = Model(
    net=unet,
    criterion=monai.losses.DiceCELoss(softmax=True),
    learning_rate=1e-2,
    optimizer_class=torch.optim.AdamW,
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)
trainer = pl.Trainer(
    gpus=1,
    precision=32,
    callbacks=[early_stopping],
)
data = MyDataModule(
    batch_size=4
)
trainer.logger._default_hp_metric = False
start = datetime.now()
print('Training started at', start)
trainer.fit(model=model, datamodule=data)
print('Training duration:', datetime.now() - start)


