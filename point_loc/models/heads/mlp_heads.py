from typing import Optional
from torch import Tensor
import torch
from mmdet3d.registry import MODELS
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union
from mmpretrain.structures import DataSample
from mmpretrain.models.heads import LinearClsHead
import torch.nn as nn
from point_loc.datasets import matrix_utils
 
@MODELS.register_module()
class LinearRegressionHead(LinearClsHead):
    
    def _get_loss(self, model_predictions: Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        assert 'gt_values' in data_samples[0], "Regression heads assume that ground truth are stored in a field named gt_values"

        target = torch.stack([i.gt_values for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            model_predictions, target, **kwargs)
        losses['loss'] = loss

        return losses

    def _get_predictions(self, model_predictions: Tensor, data_samples: List[DataSample]):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(model_prediction.size(0))]

        for data_sample, per_batch_predictions in zip(data_samples, model_predictions):
            
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_field(per_batch_predictions, 'pred_values')
            out_data_samples.append(data_sample)
        return out_data_samples
    
    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats
    
    
@MODELS.register_module()
class MLPRegressionHead(LinearRegressionHead):
    """Linear regression head with a multi-layer perceptron (MLP).

    Args:
        in_channels (int): Input channels of the MLP.
        hidden_channels (List[int]): Hidden channels of MLP layers.
        num_outputs (int): Number of outputs of the MLP.
        loss (dict): Config of regression loss. Defaults to 
        ``dict(type='MSELoss')``.
        num_shared_layers (int): Number of shared layers in MLP. Defaults to 0.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 num_outputs: int,
                 loss: dict = dict(type='MSELoss'),
                 num_shared_layers: int = 1,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        
        super().__init__(
            in_channels=in_channels,
            num_classes=num_outputs,
            loss=loss,
            init_cfg=init_cfg,
            **kwargs)

        self.num_outputs = num_outputs
        self.num_shared_layers = num_shared_layers

        # Build shared layers
        self.shared_layers = nn.ModuleList()
        for i in range(num_shared_layers):
            if i == 0:
                self.shared_layers.append(nn.Linear(in_channels, hidden_channels[i]))
            else:
                self.shared_layers.append(nn.Linear(hidden_channels[i-1], hidden_channels[i]))
            self.shared_layers.append(nn.ReLU())

        # Build specific layers for each output
        self.specific_layers = nn.ModuleList()
        for _ in range(num_outputs):
            layers = []
            for i in range(num_shared_layers, len(hidden_channels)):
                if i == 0:
                    layers.append(nn.Linear(in_channels, hidden_channels[i]))
                else:
                    layers.append(nn.Linear(hidden_channels[i-1], hidden_channels[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_channels[-1], 1))
            self.specific_layers.append(nn.Sequential(*layers))

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        x = self.pre_logits(x)
        
        # Apply shared layers
        if len(self.shared_layers) > 0:
            for layer in self.shared_layers:
                x = layer(x)

        # Apply specific layers for each output
        outputs = []
        for specific_layer in self.specific_layers:
            outputs.append(specific_layer(x))
        
        outputs = torch.cat(outputs, dim=1)
        
        L = matrix_utils.vector_to_upper_triangular_matrix(outputs)
        outputs = matrix_utils.cholesky_undecomposition(L)
        outputs = matrix_utils.symetric_matrix_to_upper_triangular_vector(outputs)
        # Concatenate outputs
        return outputs
