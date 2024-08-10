from point_loc.registry import MODELS
from mmdet.models import losses

MODELS.register_module(module=losses.CrossEntropyLoss)
MODELS.register_module(module=losses.MSELoss) 
MODELS.register_module(module=losses.L1Loss) 