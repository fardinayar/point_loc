import torch.nn.functional
from point_loc.registry import MODELS
from mmdet.models import losses
import torch
import torch.nn as nn

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
    


    
class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
        self.matrix_dim = 6

    def vector_to_symmetric_matrix(self, vec):
        batch_size = vec.size(0)
        symm_matrix = torch.zeros(batch_size, self.matrix_dim, self.matrix_dim, device=vec.device)
        indices = torch.triu_indices(self.matrix_dim, self.matrix_dim)
        symm_matrix[:, indices[0], indices[1]] = vec
        symm_matrix = symm_matrix + symm_matrix.transpose(-1, -2)
        symm_matrix[:, range(self.matrix_dim), range(self.matrix_dim)] *= 0.5
        return symm_matrix

    def forward(self, pred, gt):
        pred_cov = self.vector_to_symmetric_matrix(pred) + 1e-5
        gt_cov = self.vector_to_symmetric_matrix(gt)
        epsilon = 1e-6
        pred_cov.diagonal(dim1=-2, dim2=-1).add_(epsilon)
        gt_cov.diagonal(dim1=-2, dim2=-1).add_(epsilon)
        pred_inv = torch.inverse(pred_cov)
        gt_inv = torch.inverse(gt_cov)
        trace_term = torch.diagonal(torch.matmul(gt_inv, pred_cov), dim1=-2, dim2=-1).sum(-1)

        # The rest of the KL divergence calculation
        kl_div = 0.5 * (torch.log(torch.det(pred_cov) / torch.det(gt_cov)) - 
                        self.matrix_dim + 
                        trace_term + 
                        torch.matmul(torch.matmul((gt_inv - pred_inv), (pred_cov - gt_cov)).view(pred.size(0), -1), 
                                    torch.ones(self.matrix_dim**2, 1, device=pred.device)).squeeze())
        
        return kl_div.mean()
    
class Hardshrink(nn.Module):
    def __init__(self, lambd: float = 0.5) -> None:
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input, target):
        input = torch.abs(input - target)
        return torch.nn.functional.hardshrink(input, self.lambd)

    
MODELS.register_module(module=torch.nn.HuberLoss)
MODELS.register_module(module=Hardshrink)
MODELS.register_module(module=losses.MSELoss) 
MODELS.register_module(module=losses.L1Loss) 
MODELS.register_module(module=losses.SmoothL1Loss) 

MODELS.register_module(module=WelschLoss)
MODELS.register_module(module=GemanMcClureLoss)
MODELS.register_module(module=KLDivergenceLoss)


