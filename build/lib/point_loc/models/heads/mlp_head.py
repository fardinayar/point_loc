from typing import Dict, List, Optional, Union
from mmengine.model import BaseModel
from torch import Tensor, Tuple
from mmdet3d.utils import OptConfigType, OptMultiConfig, ConfigType
from mmdet3d.registry import MODELS
from torch import nn as nn

@MODELS.register_module()
class MLPHead(BaseModel):
    """Simple MLP head with predefined loss function for both classification and regression.

    Args:
        input_dim (int): The dim of input features.
        num_outputs (int): Number of outputs. Each output will have its own loss. Defaults to 6.
        hidden_layers_dim (int or List(int)): The dim(s) of hidden layers. Defaults to 256.
        num_hidden_layers (int): Number of hidden layers. Defaults to 2.
        loss (:obj:`ConfigDict` or dict): Config of the loss
            loss. Defaults to `CrossEntropyLoss`.
        train_cfg (:obj:`ConfigDict` or dict): Training config.
            head.
        test_cfg (:obj:`ConfigDict` or dict): Testing config.
            head.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """
    _version = 1

    def __init__(
            self,
            input_dim: int,
            num_outputs: int = 6,
            hidden_layers_dim: int = 256,
            num_hidden_layers: int = 2,
            loss: ConfigType = dict(
                type='MSELoss'),
            train_cfg: ConfigType = None,
            test_cfg: ConfigType = None,
            init_cfg: OptMultiConfig = None) -> None:
        
        super().__init__(init_cfg=init_cfg)
        
        self.num_outputs = num_outputs
        self.loss_fn = MODELS.build(loss)
    
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers_dim))
        
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_layers_dim, hidden_layers_dim))
            
        self.layers.append(nn.Linear(hidden_layers_dim, num_outputs))
        
        
    def _forward(self, 
                inputs: Tensor,
                targets: Tensor) -> Tensor:
        
        x = inputs
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def loss(self, 
            inputs: Tensor,
            targets: Tensor) -> Tensor:
        x = self._forward(inputs, targets)
        loss = self.loss_fn(x, targets)
        
        return {'loss': loss}
    
    def forward(self,
                inputs: Union[dict, List[dict]],
                targets: Tensor = None,
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
            targets (Tensor, optional):
                The annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor.
            - If ``mode="predict"``, return a tensor.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, targets)
        elif mode == 'predict':
            return self.predict(inputs, targets)
        elif mode == 'tensor':
            return self._forward(inputs, targets)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
        
