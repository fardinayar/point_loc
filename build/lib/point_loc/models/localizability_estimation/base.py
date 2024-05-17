from typing import Dict, List, Union
from mmengine.model import BaseModel
from abc import ABCMeta, abstractmethod
from torch import Tensor, Tuple
from mmdet3d.utils import OptConfigType, OptMultiConfig

class BaseEstimation(BaseModel, metaclass=ABCMeta):
    """Base class for localizability estimation.
    It can both handle classification and regression tasks depending on targets.

    Args:
        data_preprocessor (dict or ConfigDict, optional): Model preprocessing
            config for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_val``, ``mean`` and
            ``std``. Defaults to None.
       init_cfg (dict or ConfigDict, optional): The config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(BaseEstimation, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_auxiliary_head(self) -> bool:
        """bool: Whether the estimation head has auxiliary head."""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None
    
    @property
    def with_neck(self) -> bool:
        """bool: Whether the estimation head has auxiliary head."""
        return hasattr(self,
                       'neck') and self.neck is not None  
        
    @property
    def with_voxel_encoder(self) -> bool:
        """bool: Whether the estimation head has auxiliary head."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None    
    
    @property
    def with_regularization_loss(self) -> bool:
        """bool: Whether the estimation head has regularization loss for weight."""
        return hasattr(self, 'loss_regularization') and \
            self.loss_regularization is not None
            
    @property
    def with_decoder(self) -> bool:
        """bool: Whether the estimation head has regularization loss for weight."""
        return hasattr(self, 'loss_regularization') and \
            self.loss_regularization is not None
            
    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor) -> dict:
        """Placeholder for extract features from points."""
        pass
    
    
    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: List[Tensor] = None,
                mode: str = 'tensor') -> Union[Dict[str, Tensor], Tensor]:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "predict", "tensor" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (dict or List[dict]): Input sample dict which includes
                'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor): Image tensor has shape (B, C, H, W).
            data_samples (List[Tensor], optional):
                The annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor.
            - If ``mode="predict"``, return a tensor.
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
             batch_data_targets: List[Tensor]) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and targets."""
        pass
    
    @abstractmethod
    def predict(self, batch_inputs: dict,
                batch_data_targets: List[Tensor] = None) -> List[Tensor]:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass
    
    @abstractmethod
    def _forward(self,
                 batch_inputs: dict,
                 batch_data_targets: List[Tensor] = None) -> Tensor:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass
    
