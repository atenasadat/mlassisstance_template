import torch
from torch import nn
import torch.nn.functional as F
from mlassistant.core import ModelIO, Model


class MNISTClassifier(Model):
    def __init__(self):
        super().__init__()
        
        self._seq = nn.Sequential(                          # B 1   28  28
            nn.BatchNorm2d(num_features=1),
            nn.Conv2d(1, 64, 3, stride=1, padding=3),       # B 64  32  32  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),      # B 64  32  32
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),           # B 64  16  16
            nn.Dropout2d(0.2, inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),     # B 128 16  16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),    # B 128 16  16
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),           # B 128 8   8
            nn.Dropout2d(0.2, inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),    # B 256 8   8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),    # B 256 8   8
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),           # B 256 4   4
            nn.Dropout2d(0.2, inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),    # B 512 4   4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),    # B 512 4   4
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),           # B 512 2   2
            nn.Dropout2d(0.2, inplace=True),
            nn.Flatten(1),                                  # B 2048
            nn.Linear(2048, 512),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
            nn.Softmax(dim=-1))
    
    def forward(self, mnist_x: torch.Tensor, mnist_y: torch.Tensor) -> ModelIO:
        # x:    B   1   28  28
        out = self._seq(mnist_x)

        output = {
            'categorical_probability': out,
        }

        if mnist_y is not None:
            output['loss'] = F.cross_entropy(out, mnist_y.long())
        
        return output
