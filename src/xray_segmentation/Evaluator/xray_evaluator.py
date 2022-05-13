
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Dict, TYPE_CHECKING
import numpy as np
from torch import Tensor
from mlassistant.core.data import DataLoader
from mlassistant.core import Model
from mlassistant.model_evaluation import NormalEvaluator
if TYPE_CHECKING:
    from ..configs import XRayConfig


# def dice_coef(y_true, y_pred):
#     y_true_f = torch.flatten(y_true)
#     y_pred_f = torch.flatten(y_pred)
#     print("shapes" , y_true_f.shape , y_pred_f.shape)
#     intersection = torch.sum(y_true_f * y_pred_f)
#     return (2. * intersection + 1) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + 1)

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)


class SegEvaluator(NormalEvaluator):

    def __init__(self, model: Model, data_loader: DataLoader, conf: 'XRayConfig'):
        super(SegEvaluator, self).__init__(model, data_loader, conf)

        self.avg_loss: float = 0.0
        self.mean_sample_loss: float = 0.0
        self.max_sample_loss: float = 0.0
        self.avg_sample_sqr_loss: float = 0.0
        self.n_received_samples: float = 0

        self.sample_sum_loss = np.zeros((data_loader.get_number_of_samples(),), dtype=float)
        self.sample_n_repeats = np.zeros((data_loader.get_number_of_samples(),), dtype=float)

    def reset(self):
        self.avg_loss = 0.0
        self.mean_sample_loss = 0.0
        self.max_sample_loss = 0.0
        self.avg_sample_sqr_loss = 0.0
        self.n_received_samples = 0

        self.sample_sum_loss = np.zeros((self.data_loader.get_number_of_samples(),),
                                        dtype=float)
        self.sample_n_repeats = np.zeros((self.data_loader.get_number_of_samples(),),
                                         dtype=float)



    def update_summaries_based_on_model_output(self, model_output: Dict[str, Tensor]) -> None:



        losses = model_output['loss'].cpu().numpy()

        n_new = len(losses) + self.n_received_samples
        f_old = float(self.n_received_samples) / n_new
        f_new = float(len(losses)) / n_new

        self.mean_sample_loss = f_old * self.mean_sample_loss + f_new * np.mean(losses)
        self.max_sample_loss = max(self.max_sample_loss, np.amax(losses))
        self.avg_sample_sqr_loss = f_old * self.avg_sample_sqr_loss + \
                                   f_new * np.mean(losses ** 2)
        self.avg_loss = f_old * self.avg_loss + f_new * model_output.get('loss', 0.0)

        self.n_received_samples = n_new

    def update_samples_related_details_based_on_model_output(self, model_output: Dict[str, Tensor]) -> None:


        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()

        sample_loss = model_output['loss'].cpu().numpy()

        np.add.at(self.sample_sum_loss, current_batch_sample_indices, sample_loss)
        np.add.at(self.sample_n_repeats, current_batch_sample_indices, 1)



    def save_middle_outputs_of_the_model(self, model_output: Dict[str, Tensor], save_dir: str) -> None:
        raise NotImplementedError('')


    def get_samples_results_summaries(self):
        samples_names = self.data_loader.get_samples_names()
        return samples_names, ['%d\t%.2f' %
                               (self.sample_n_repeats[i],
                                self.sample_sum_loss[i] / self.sample_n_repeats[i])
                               for i in range(len(samples_names))]

    def get_samples_results_header(self):
        return ['N_Els', 'AverageLoss']

    def get_samples_elements_results_summaries(self):
        raise Exception('Not applicable to this class')

    def get_titles_of_evaluation_metrics(self) -> List[str]:
        return ['Loss', 'MeanSLoss', 'StdSLoss', 'MaxSLoss']


    def get_values_of_evaluation_metrics(self) -> List[str]:
        return ['%.4f' % self.avg_loss, '%.4f' % self.mean_sample_loss,
                '%.4f' % (self.avg_sample_sqr_loss - self.mean_sample_loss ** 2),
                '%.4f' % self.max_sample_loss]


class DiceCoefficient(torch.autograd.Function):
    """
    Dice coefficient for individual examples
        Dice coefficient = 2 * |X n Y| / (|X| + |Y|)
                         = 1 / ( 1/Precision + 1/Recall)
    """
    def forward(self, input , target):
        self.save_for_backward(input, target)
        eps = 1e-10
        self.inter = torch.dot(input.view(-1), target.view(-1)) # inter = |X n Y|
        self.union = torch.sum(input) + torch.sum(target) # union =|X| + |Y|

        dice = 2 * self.inter.float() / (self.union.float() + eps)
        return dice

    def backward(self ,grad_output):

        input ,target = self.saved_variables
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / ( self.union * self.union )

        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coef_loss(input, target):
    """Dice coeff for batches"""
    s = torch.FloatTensor(1).cuda().zero_()
    # else:
    #     s = torch.FloatTensor(1).zero_()

    for i , c in enumerate(zip(input, target)):
        s += DiceCoefficient().forward(c[0] ,c[1])

    n_data = len(input)
    return s / n_data
