from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.models.utils import add_prefix
from .base import BaseEstimation

@MODELS.register_module()
class SimpleEncoder(BaseEstimation):
    """Simple encoder for localizability estimation.

    Encoder typically consists of backbone, auxiliary_head and a head for classification/regression.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the head and loss function to calculate losses.

    .. code:: text

    loss(): extract_feat() -> _head_forward_train()
                              _auxiliary_head_forward_train (optional)
    _head_forward_train(): head.loss()
    _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict localizability results,
    which run inference function to obtain the list of seg_logits 

    .. code:: text

    predict(): inference()
    inference(): whole_inference()
    whole_inference(): extract_feat() -> head.predict()

    3 The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the head forward function to forward head model.

    .. code:: text

    _forward(): extract_feat() -> head.forward()

    Args:
        voxel_encoder (dict or :obj:`ConfigDict`): The config for the
            points2voxel encoder of segmentor.
        backbone (dict or :obj:`ConfigDict`): The config for the backbone.
        head (dict or :obj:`ConfigDict`): The config for the decode
            head.
        neck (dict or :obj:`ConfigDict`, optional): The config for the neck of
            segmenter. Defaults to None.
        auxiliary_head (dict or :obj:`ConfigDict` or List[dict or
            :obj:`ConfigDict`], optional): The config for the auxiliary head of.
            Defaults to None.
        loss_regularization (dict or :obj:`ConfigDict` or List[dict or
            :obj:`ConfigDict`], optional): The config for the regularization
            losses. Defaults to None.
        train_cfg (dict or :obj:`ConfigDict`, optional): The config for
            training. Defaults to None.
        test_cfg (dict or :obj:`ConfigDict`, optional): The config for testing.
            Defaults to None.
        data_preprocessor (dict or :obj:`ConfigDict`, optional): The
            pre-process config of :class:`BaseDataPreprocessor`.
            Defaults to None.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`],
            optional): The weight initialized config for :class:`BaseModule`.
            Defaults to None.
    """  # noqa: E501
    
    def __init__(self,
                 backbone: ConfigType,
                 head: ConfigType,
                 voxel_encoder: ConfigType = None,
                 neck: OptConfigType = None,
                 auxiliary_head: OptMultiConfig = None,
                 loss_regularization: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(SimpleEncoder, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if voxel_encoder is not None:
            self.voxel_encoder = MODELS.build(voxel_encoder)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.head = MODELS.build(head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_loss_regularization(loss_regularization)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
    def _init_auxiliary_head(self,
                             auxiliary_head: OptMultiConfig = None) -> None:
        """Initialize ``auxiliary_head``."""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)
                
    def _init_loss_regularization(self,
                                  loss_regularization: OptMultiConfig = None
                                  ) -> None:
        """Initialize ``loss_regularization``."""
        if loss_regularization is not None:
            if isinstance(loss_regularization, list):
                self.loss_regularization = nn.ModuleList()
                for loss_cfg in loss_regularization:
                    self.loss_regularization.append(MODELS.build(loss_cfg))
            else:
                self.loss_regularization = MODELS.build(loss_regularization)

    
    def extract_feat(self, batch_inputs: Dict[str, Tensor]) -> dict:
        """Extract features from points or voxels."""
        if self.with_voxel_encoder:
            encoded_feats = self.voxel_encoder(batch_inputs['voxels']['voxels'],
                                           batch_inputs['voxels']['coors'])
            batch_inputs['voxels']['voxel_coors'] = encoded_feats[1]
            x = self.backbone(encoded_feats[0], encoded_feats[1],
                            len(batch_inputs['points']))
        else:
            x = self.backbone(torch.stack(batch_inputs['points']))
        if self.with_neck:
            x = self.neck(x)
        return x
    
    
    def _head_forward_train(
            self, batch_inputs_dict: dict,
            batch_data_targets: List[Tensor]) -> Dict[str, Tensor]:
        """Run forward function and calculate loss for decode head in training.

        Args:
            batch_input (Tensor): Input point clouds
            batch_data_targets (Tensor): Targets

        Returns:
            Dict[str, Tensor]: A dictionary of loss components for decode head.
        """
        losses = dict()
        loss = self.head.loss(batch_inputs_dict,
                              batch_data_targets)

        losses.update(add_prefix(loss, 'head'))
        return losses
    
    def _auxiliary_head_forward_train(
        self,
        batch_inputs_dict: Tensor,
        batch_data_targets: Tensor,
    ) -> Dict[str, Tensor]:
        """Run forward function and calculate loss for auxiliary head in
        training.

        Args:
            batch_input (Tensor): Input point cloud sample
            batch_data_targets (List[:obj:`Tensor`]): Targets
        Returns:
            Dict[str, Tensor]: A dictionary of loss components for auxiliary
            head.
        """
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(batch_inputs_dict, batch_data_targets,
                                         self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(batch_inputs_dict,
                                                batch_data_targets,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses
    
    def _loss_regularization_forward_train(self) -> Dict[str, Tensor]:
        """Calculate regularization loss for model weight in training."""
        losses = dict()
        if isinstance(self.loss_regularization, nn.ModuleList):
            for idx, regularize_loss in enumerate(self.loss_regularization):
                loss_regularize = dict(
                    loss_regularize=regularize_loss(self.modules()))
                losses.update(add_prefix(loss_regularize, f'regularize_{idx}'))
        else:
            loss_regularize = dict(
                loss_regularize=self.loss_regularization(self.modules()))
            losses.update(add_prefix(loss_regularize, 'regularize'))

        return losses
    
    def loss(self, batch_inputs_dict: Dict[str, Tensor],
             batch_data_targets: List[Tensor]) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_targets (List[Tensor]): Targets

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        x = self.extract_feat(batch_inputs_dict)
        batch_data_targets = torch.stack(batch_data_targets)
        losses = dict()
        loss_decode = self._head_forward_train(x, batch_data_targets)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, batch_data_targets)
            losses.update(loss_aux)

        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses
    
    
    def predict(self,
                batch_inputs_dict: dict,
                batch_data_targets: List[Tensor]) -> List[Tensor]:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_targets ([:obj:`Tensor`]): Targets.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            Tensor: Segmentation results of the input
            points. 
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        loc_logits_list = []

        points = batch_inputs_dict
        if 'voxels' in points:
            loc_logits = self._forward(
                    points)
            return loc_logits
        else:
            for i in range(len(points['points'])):
                input_dict = {k: [batch_inputs_dict[k][i]] for k in batch_inputs_dict.keys()}
                loc_logits = self._forward(
                    input_dict)
                loc_logits_list.append(loc_logits)
            return torch.cat(loc_logits_list)

    def _forward(self,
                 batch_inputs_dict: Dict[str, Tensor],
                 batch_data_targets: List[Tensor] = None) -> List[Tensor]:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_targets (Tensor): Targets.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(batch_inputs_dict)
        return self.head.forward(x)