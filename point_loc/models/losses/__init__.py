import torch.nn.functional
from point_loc.registry import MODELS
from mmdet.models import losses
import torch
import torch.nn as nn
from point_loc.datasets import matrix_utils

class WelschLoss(torch.nn.Module):
    def __init__(self):
        super(WelschLoss, self).__init__()
    
    def forward(self, x, y):
        return 1 - torch.exp(-0.5 * torch.abs(x - y))
    

class GemanMcClureLoss(torch.nn.Module):
    def __init__(self):
        super(GemanMcClureLoss, self).__init__()
    
    def forward(self, x, y):
        diff = x - y
        return 2 * (diff ** 2) / (diff ** 2 + 4)

class HardShrink(torch.nn.Module):
    def __init__(self, beta=0.01):
        super(HardShrink, self).__init__()
        self.beta = beta
        
    def forward(self, target, pred):
        x = torch.abs(target - pred)
        return torch.nn.functional.hardshrink(x, self.beta)
    
    
class KLDivergenceLoss(torch.nn.Module):
    
    def __init__(self,
                 ) -> None:
        super().__init__()
        

    def forward(
        self,target, pred
    ) -> torch.Tensor:
        """Calculate the Mean Absolute Error (MAE) for each dimension.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction covariance matrix as upper traingular vector matrix.
            target (torch.Tensor | np.ndarray | Sequence): The target covariance matrix as upper traingular vector matrix..

        """
        
        # iterate over pred-target pairs
        l1 = torch.nn.functional.huber_loss(pred, target, reduction='mean')
        
        pred = matrix_utils.vector_to_symmetric_matrix(pred) + torch.eye(6).cuda() * 1e-5  # small constant
        target = matrix_utils.vector_to_symmetric_matrix(target) + torch.eye(6).cuda() * 1e-5  # small constant
        kl = torch.distributions.kl.kl_divergence(
            torch.distributions.MultivariateNormal(loc=torch.zeros((pred.shape[0],6)).cuda(), covariance_matrix=pred),
            torch.distributions.MultivariateNormal(loc=torch.zeros((pred.shape[0],6)).cuda(), covariance_matrix=target)
        )
            
        return kl.mean() + l1.mean()
        
        

        

MODELS.register_module(module=losses.CrossEntropyLoss)
MODELS.register_module(module=losses.MSELoss) 
MODELS.register_module(module=losses.L1Loss) 
MODELS.register_module(module=losses.SmoothL1Loss) 
MODELS.register_module(module=torch.nn.HuberLoss) 
MODELS.register_module(module=HardShrink) 

MODELS.register_module(module=WelschLoss)
MODELS.register_module(module=GemanMcClureLoss)
MODELS.register_module(module=KLDivergenceLoss)



