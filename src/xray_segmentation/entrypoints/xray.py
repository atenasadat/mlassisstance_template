from mlassistant.entrypoint import BaseEntryPoint
from xray_segmentation.models.xray_unet import UNet
from ..config import XRayConfig

import os

path = os.getcwd()


class EntryPoint(BaseEntryPoint):
    r'''The name of this class **MUST** be `EntryPoint`'''
    def __init__(self):
        super().__init__(XRayConfig(try_name='Xray', try_num=10),
                         UNet(n_channels = 3 , n_classes =3 ))
