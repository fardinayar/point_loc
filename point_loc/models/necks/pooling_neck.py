# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.registry import MODELS
from typing import Dict, List, Optional, Union
import torch
from mmcv.ops.sparse_modules import SparseConvTensor

@MODELS.register_module()
class PoolingNeck(BaseModule):
    r"""A simple pooling without parameters to aggregate features of last layer of backbones.
        In mmdet3d backbones usually returns a dict or a list of multiscale features, so here
        we use the last (the most abstract) features for pooling.

    Args:
        feature_name (str): If the features from the backbone are in a dict, for example
            features from PointNet2, this name will be used as key to access the features.
        pool_mod (str): The pooling mode. Defaults to avg.
        point_dim (int): The dim of points or voxels in the feature tensors, for example,
            in some backbones, features are of the dimension (B, N, C) which N is the number of points.
            Defaults to -1, which means that features are of the dimension (B, C, N).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, feature_name=None, point_dim=1, pool_mod='avg', init_cfg=None):
        super(PoolingNeck, self).__init__(init_cfg=init_cfg)

        self.feature_name = feature_name
        if pool_mod == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_mod == 'max':
            self.pool == nn.AdaptiveMaxPool1d(1)
        else:
            raise TypeError('pool_mod should be avg or max')
        self.point_dim = point_dim
        assert point_dim == 1, 'point_dim other than 1 is not considered yet.'

    def forward(self, backbone_features):
        """Pool backbone features to be passed to an MLP head.

        Args:
            feat_dict Union(dict, list, torch.Tensor): Torch.Tensor or List or feature dict from backbone,
                which may contain different keys and values.

        Returns:
            torch.Tensor: Features of last level of backbone.
        """

        if type(backbone_features) == list:
            return self.pool(backbone_features[-1]).squeeze(-1)
        elif type(backbone_features) == dict:
            backbone_features = backbone_features[self.feature_name]
            return self.forward(backbone_features)
        elif type(backbone_features) == torch.Tensor:
            return self.pool(backbone_features).squeeze(-1)
        elif type(backbone_features) == SparseConvTensor:
            return self.pool(backbone_features.dense().flatten(2)).squeeze(-1)
