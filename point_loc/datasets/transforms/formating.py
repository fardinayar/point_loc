# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import List, Sequence, Union

import numpy as np
import torch
from mmcv import BaseTransform
from mmdet3d.structures.points import BasePoints
from point_loc.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import to_tensor
from mmpretrain.structures import DataSample, MultiTaskDataSample

@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'img']

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global',
                            'axis_align_matrix')
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError
        
    def format_input(self, results: dict):
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img
                
        
        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                    
        return inputs
        

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        
        inputs = self.format_input(results)


        packed_results = dict()
        data_sample = DataSample()
        if 'gt_label' in results:
            data_sample.set_gt_label(results['gt_label'])
        if 'gt_score' in results:
            data_sample.set_gt_score(results['gt_score'])
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


# From mmpretrain
@TRANSFORMS.register_module()
class PackMultiTaskInputs(BaseTransform):
    """Convert all image labels of multi-task dataset to a dict of tensor.

    Args:
        multi_task_fields (Sequence[str]):
        input_keys (Sequence[str]):
        task_handlers (dict):
    """

    def __init__(self,
                 multi_task_fields,
                 task_handlers=dict(),
                 ):
        self.multi_task_fields = multi_task_fields
        self.task_handlers = task_handlers
        for task_name, task_handler in task_handlers.items():
            self.task_handlers[task_name] = TRANSFORMS.build(task_handler)

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        result = {'img_path': 'a.png', 'gt_label': {'task1': 1, 'task3': 3},
            'img': array([[[  0,   0,   0])
        """
        results = results.copy()

        
        inputs = next(iter(self.task_handlers.values())).format_input(results)

        task_results = defaultdict(dict)
        for field in self.multi_task_fields:
            if field in results:
                value = results.pop(field)
                for k, v in value.items():
                    task_results[k].update({field: v})

        data_sample = MultiTaskDataSample()
        for task_name, task_result in task_results.items():
            task_handler = self.task_handlers[task_name]
            task_pack_result = task_handler({**task_result})
            data_sample.set_field(task_pack_result['data_samples'], task_name)

        packed_results = dict()
        packed_results['inputs'] = inputs
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self):
        repr = self.__class__.__name__
        task_handlers = ', '.join(
            f"'{name}': {handler.__class__.__name__}"
            for name, handler in self.task_handlers.items())
        repr += f'(multi_task_fields={self.multi_task_fields}, '
        repr += f"input_key='{self.input_key}', "
        repr += f'task_handlers={{{task_handlers}}})'
        return repr