from point_loc.registry import MODELS
from mmpretrain.models import heads
from .mlp_heads import LinearRegressionHead

__all__ = ['LinearRegressionHead']

MODELS.register_module(module=heads.LinearClsHead)