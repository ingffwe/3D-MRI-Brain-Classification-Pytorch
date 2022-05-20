'''
unet(only downsampling)
'''
import torch
import torch.nn as nn
from collections import OrderedDict


class Unet(nn.Module):
    def __init__(self, n_classes=2, input_shape=(64,64,64)):
        super(Unet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape

        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=3, stride=1)),
            ('relu1', torch.nn.ReLU()),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool1', torch.nn.MaxPool3d(2)),
            ('conv3d_3', torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)),
            ('relu3', torch.nn.ReLU()),
            ('conv3d_4', torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)),
            ('relu4', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('conv3d_5', torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3)),
            ('relu5', torch.nn.ReLU()),
            ('conv3d_6', torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3)),
            ('relu6', torch.nn.ReLU()),
            ('pool3', torch.nn.MaxPool3d(2)),
            ('conv3d_7', torch.nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3)),
            ('relu5', torch.nn.ReLU()),
            # ('conv3d_8', torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3)),
            # ('relu6', torch.nn.ReLU()),

        ]))

        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(128, self.n_classes))
        ]))

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    unet = Unet()
    print(unet)
    data = torch.rand([8, 1, 64, 64, 64])
    unet(data)