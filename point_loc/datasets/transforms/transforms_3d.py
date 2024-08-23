# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import GlobalRotScaleTrans as mmdet3d_GlobalRotScaleTrans
from point_loc.registry import TRANSFORMS
from point_loc.datasets import matrix_utils
from .utils import transform_covariance
from typing import List

@TRANSFORMS.register_module()
class GlobalRotScaleTransWithCov(mmdet3d_GlobalRotScaleTrans):
    """Apply global rotation, scaling and translation to a 3D scene.

    Required Keys:

    - points (np.float32)
    - gt_values (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_values (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of
            translation noise applied to a scene, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range: List[float] = [-0.78539816, 0.78539816],
                 scale_ratio_range: List[float] = [0.95, 1.05],
                 translation_std: List[int] = [0, 0, 0],
                 shift_height: bool = False) -> None:
        super(GlobalRotScaleTransWithCov, self).__init__(rot_range, scale_ratio_range, translation_std, shift_height)

    def _trans_covariances_points(self, input_dict: dict) -> None:
        """Private function to translate covariances and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
            and `gt_values` is updated in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor

    def _rot_covariances_points(self, input_dict: dict) -> None:
        """Private function to rotate covariances and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
            and `gt_values` is updated in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])
            
        rot_mat_T = input_dict['points'].rotate(noise_rotation)
          
        if 'gt_values' in input_dict:
            covariance = matrix_utils.vector_to_symmetric_matrix(input_dict['gt_values'])
            covariance = transform_covariance(covariance, rot_mat_T.numpy())
            input_dict['gt_values'] = matrix_utils.symetric_matrix_to_upper_triangular_vector(covariance)

        
        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = noise_rotation

    def _scale_bbox_points(self, input_dict: dict) -> None:
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points' and
            `gt_bboxes_3d` is updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].scale(scale)

    def _random_scale(self, input_dict: dict) -> None:
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor'
            are updated in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_covariances_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_covariances_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str

