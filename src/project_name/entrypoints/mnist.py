from mlassistant.entrypoint import BaseEntryPoint
from ..models.mnist_classifier import MNISTClassifier
from ..config import MnistConfig


class EntryPoint(BaseEntryPoint):
    r'''The name of this class **MUST** be `EntryPoint`'''

    def __init__(self):
        super().__init__(MnistConfig(try_name='Mnist', try_num=1),
                         MNISTClassifier())
