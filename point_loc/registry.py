# Copyright (c) OpenMMLab. All rights reserved.
"""MMDetection3D provides 17 registry nodes to support using modules across
projects. Each node is a child of the root registry in mmdet3d.

More details can be found at
https://mmdet3d.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmdet3d.registry import DATA_SAMPLERS as mmdet3d_DATA_SAMPLERS
from mmdet3d.registry import DATASETS as mmdet3d_DATASETS
from mmdet3d.registry import EVALUATOR as mmdet3d_EVALUATOR
from mmdet3d.registry import HOOKS as mmdet3d_HOOKS
from mmdet3d.registry import INFERENCERS as mmdet3d_INFERENCERS
from mmdet3d.registry import LOG_PROCESSORS as mmdet3d_LOG_PROCESSORS
from mmdet3d.registry import LOOPS as mmdet3d_LOOPS
from mmpretrain.registry import METRICS as mmpretrain_METRICS
from mmdet3d.registry import MODEL_WRAPPERS as mmdet3d_MODEL_WRAPPERS
from mmdet3d.registry import MODELS as mmdet3d_MODELS
from mmdet3d.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as mmdet3d_OPTIM_WRAPPER_CONSTRUCTORS
from mmdet3d.registry import OPTIM_WRAPPERS as mmdet3d_OPTIM_WRAPPERS
from mmdet3d.registry import OPTIMIZERS as mmdet3d_OPTIMIZERS
from mmdet3d.registry import PARAM_SCHEDULERS as mmdet3d_PARAM_SCHEDULERS
from mmdet3d.registry import \
    RUNNER_CONSTRUCTORS as mmdet3d_RUNNER_CONSTRUCTORS
from mmdet3d.registry import RUNNERS as mmdet3d_RUNNERS
from mmdet3d.registry import TASK_UTILS as mmdet3d_TASK_UTILS
from mmdet3d.registry import TRANSFORMS as mmdet3d_TRANSFORMS
from mmdet3d.registry import VISBACKENDS as mmdet3d_VISBACKENDS
from mmdet3d.registry import VISUALIZERS as mmdet3d_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as mmdet3d_WEIGHT_INITIALIZERS
from mmengine import Registry
import mmdet3d.registry

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=mmdet3d_DATASETS, locations=['point_loc.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=mmdet3d_DATA_SAMPLERS,
    # TODO: update the location when point_loc has its own data sampler
    locations=['point_loc.datasets'])

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', parent=mmdet3d_MODELS, locations=['point_loc.models'])
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=mmdet3d_MODEL_WRAPPERS,
    locations=['point_loc.models'])
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=mmdet3d_WEIGHT_INITIALIZERS,
    locations=['point_loc.models'])
# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=mmdet3d_TASK_UTILS, locations=['point_loc.models'])

TRANSFORMS = Registry(
    'transform',
    parent=mmdet3d_TRANSFORMS, locations=['point_loc.datasets.transforms'])

METRICS = Registry(
    'metric',
    parent=mmpretrain_METRICS, locations=['point_loc.evaluation']
)