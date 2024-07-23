from typing import Optional
from torch import Tensor
from mmdet3d.registry import MODELS
from mmpretrain.models import LinearClsHead as mmpretrain_LinearClsHead

@MODELS.register_module()
class LinearClsHead(mmpretrain_LinearClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg, **kwargs)
        
    def pre_logits(self, feats: Tensor) -> Tensor:
        return feats
        
