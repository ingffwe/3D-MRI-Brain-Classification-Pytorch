import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from dataloader_deprecated import Split_Data, CustomImageDataset
from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, EnsureType
import matplotlib.pyplot as plt
import models.c3d

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

data_path = r'D:\3DCLFNii\txt'
train_txt_path = data_path + "\\train_fALFF.txt"
val_txt_path = data_path + "\\val_fALFF.txt"

train_data, train_label, valid_data, valid_label = Split_Data(train_txt_path, val_txt_path, seed=1)
print(train_data[:10], train_label[:10])

# loading data
train_loader = DataLoader(CustomImageDataset(train_data, train_label, in_channels=3,reshape_size=(64,112,112)), batch_size=2, shuffle=True)
val_loader = DataLoader(CustomImageDataset(valid_data, valid_label, in_channels=3,reshape_size=(64,112,112)), batch_size=2, shuffle=False)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model
# model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
# model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)

# Unet
# model = monai.networks.nets.unet(spatial_dims=3, in_channels=1, out_channels=1, channels=(8, 16, 32, 64), strides=(2, 2,2)).to(device)


# ViT
# model = monai.networks.nets.ViT(in_channels=1, img_size=(64,64,64), patch_size=(16,16,16), pos_embed='conv',classification=True,num_classes=2)
# model = model.cuda()

# self-designed models
model = models.c3d.get_model(sample_size=112, sample_duration=64, num_classes=2)
model = model.cuda()


# parameter
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
epochs = 50
# start a typical PyTorch training
val_interval = 2
best_metric = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()
for epoch in range(epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # one_hot = torch.zeros(np.array(batch_size, num_class, device=torch.device('cuda:0')).scatter_(1, label, 1)
        labels = labels.to(torch.int64)
        # outputs = torch.tensor([item.cpu().detach().numpy() for item in outputs]).cuda()


        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_data) // train_loader.batch_size
        # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
            metric = num_correct / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_accuracy", metric, epoch + 1)
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()