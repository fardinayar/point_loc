from typing import Optional
from torch import Tensor
import torch
from mmdet3d.registry import MODELS
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union
from mmpretrain.structures import DataSample
from mmpretrain.models.heads import LinearClsHead
    
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