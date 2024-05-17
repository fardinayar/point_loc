# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Union
import mmengine
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type

