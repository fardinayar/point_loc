from point_loc.registry import MODELS
from mmdet.models import losses
import torch

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
    
MODELS.register_module(module=losses.CrossEntropyLoss)
MODELS.register_module(module=losses.MSELoss) 
MODELS.register_module(module=losses.L1Loss) 
MODELS.register_module(module=losses.SmoothL1Loss) 

MODELS.register_module(module=WelschLoss)
MODELS.register_module(module=GemanMcClureLoss)



