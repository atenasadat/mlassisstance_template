from torch import nn
from torchvision import transforms
from mlassistant.config import NormalConfig
from mlassistant.model_evaluation.multiclass_evaluator import MulticlassEvaluator
from ..data import MnistLoader


class MnistConfig(NormalConfig):

    def __init__(self, try_name: str, try_num: int):

        super().__init__(
            data_separation=None,
            try_name=try_name,
            try_num=try_num,
            evaluator_cls=MulticlassEvaluator,
            content_loaders=[('mnist', MnistLoader)],
            inp_size=28
        )

        # replaced configs!
        self.batch_size = 512
        self.training_config.iters_per_epoch = None

        # augmentation
        self.training_config.augmentations_dict = {
            'mnist_x': nn.Sequential(
                transforms.RandomRotation(35),
                transforms.RandomAffine(0, shear=0.2, scale=(.8, 1.2)),
                transforms.RandomResizedCrop(self.inp_size, scale=(0.7, 1.4)),
                transforms.RandomPerspective())
        }
