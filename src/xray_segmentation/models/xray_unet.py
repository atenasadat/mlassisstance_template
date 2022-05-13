import torch
from torch import nn
import torch.nn.functional as F
from mlassistant.core import ModelIO, Model
from xray_segmentation.Evaluator.xray_evaluator import dice_coef_loss


class UNet(Model):

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.in_conv = UNetConvBlock(self.n_channels, 64)
        self.Down1 = Down(64, 128)
        self.Down2 = Down(128, 256)
        self.Down3 = Down(256, 512)
        self.Down4 = Down(512, 512)
        self.Up1 = Up(512 + 512, 256, self.bilinear)
        self.Up2 = Up(256 + 256, 128, self.bilinear)
        self.Up3 = Up(128 + 128, 64, self.bilinear)
        self.Up4 = Up(64 + 64, 64, self.bilinear)
        self.out_conv = OutConv(64, n_classes)

    def forward(self, xray_x, xray_mask):
        # print("mnist_x", mnist_x.shape , mnist_mask.shape)

        x1 = self.in_conv(xray_x)
        # print("x1" , x1.shape)
        x2 = self.Down1(x1)
        # print("x2" , x2.shape)
        x3 = self.Down2(x2)
        # print("x3" , x3.shape)
        x4 = self.Down3(x3)
        # print("x4" , x4.shape)
        x5 = self.Down4(x4)
        # print("x5" , x5.shape)
        x = self.Up1(x5, x4)
        # print("x" , x.shape)
        x = self.Up2(x, x3)
        # print("x" , x.shape)
        x = self.Up3(x, x2)
        # print("x7" , x.shape)
        x = self.Up4(x, x1)
        # print("x8" , x.shape)
        out = self.out_conv(x)
        # print("out" , out.shape)

        output = {

            'output': out
        }

        output['loss'] = dice_coef_loss(xray_mask, out)

        return output


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Usually Conv -> BatchNormalization -> Activation
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=int(padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, inp):
        return self.double_conv(inp)


class Down(nn.Module):
    """
    Downscaling with maxpool and then Double Conv
        - 3x3 Conv2D -> BN -> ReLU
        - 3X3 Conv2D -> BN -> ReLU
        - MaxPooling
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()

        if bilinear:  # use the normal conv to reduce the number of channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:  # use Transpose convolution (the one that official UNet used)
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = UNetConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # input dim is CHW
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

