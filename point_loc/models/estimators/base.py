from typing import Dict, List, Union
from mmengine.model import BaseModel
from abc import ABCMeta, abstractmethod
from torch import Tensor, Tuple
from mmdet3d.utils import OptConfigType, OptMultiConfig
from mmpretrain.structures import DataSample

class BaseEstimator(BaseModel, metaclass=ABCMeta):
    """
    Point Loc library is developed for scene-level estimation tasks, like point cloud classification in the simplest case.
    and BaseEstimator class is the base class for all of them.
    we assume an Estimator has 4 parts:
        0) Voxel encoder (optional) that takes raw lidar points and produces voxels for voxel-based models. 
        1) Backbone that gets inputs and extracts features.
        2) Decoder (Optional) that takes input from the backbone and upsamples them. 
            Note: having a decoder to upsample features is not common in scene-level tasks, but for now, we define it to exist 
            as some papers have it.
        3) Neck that takes features from the past components and makes them ready for the head. For example, it applies global average pooling.
        4) Heads that produce scene-level logits. The estimator can have auxiliary heads.
    Adapted From <https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/segmentors/base.py>.

    Args:
        data_preprocessor (dict or ConfigDict, optional): Model preprocessing
            config for processing the input data. 
       init_cfg (dict or ConfigDict, optional): The config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(BaseEstimator, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_auxiliary_head(self) -> bool:
        """Whether the estimator has an auxiliary head."""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None
    
    @property
    def with_neck(self) -> bool:
        """Whether the estimator has a neck."""
        return hasattr(self,
                       'neck') and self.neck is not None  
        
    @property
    def with_voxel_encoder(self) -> bool:
        """Whether the estimator has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None    
    
    @property
    def with_regularization_loss(self) -> bool:
        """Whether the estimator has regularization loss for weight."""
        return hasattr(self, 'loss_regularization') and \
            self.loss_regularization is not None
            
    @property
    def with_decoder(self) -> bool:
        """Whether the estimator has a decoder."""
        return hasattr(self, 'loss_regularization') and \
            self.loss_regularization is not None
            
    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor) -> dict:
        """Placeholder to extract features from points."""
        pass
    
    
    def forward(self,
                inputs: dict,
                data_samples: List[DataSample] = None,
                mode: str = 'tensor') -> Union[Dict[str, Tensor], Tensor, List[DataSample]]:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "predict", "tensor" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (dict): Input sample dict which includes
                'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor): Image tensor has shape (B, C, H, W).
            data_samples (List[Tensor], optional):
                The annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor.
            - If ``mode="predict"``, return a list of `DataSample`s.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
        
    @abstractmethod
    def loss(self, batch_inputs: dict,
             batch_data_targets: List[DataSample]) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and targets."""
        pass
    
    @abstractmethod
    def predict(self, batch_inputs: dict,
                batch_data_targets: List[DataSample] = None) -> List[DataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass
    
    @abstractmethod
    def _forward(self,
                 batch_inputs: dict,
                 batch_data_targets: List[DataSample] = None) -> Tensor:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass
    
