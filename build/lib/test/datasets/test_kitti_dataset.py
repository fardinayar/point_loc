# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from mmdet3d.registry import TRANSFORMS
from point_loc.datasets import LocDataset
import unittest
from mmengine import DefaultScope

class TestKittiDataset(unittest.TestCase):
    def _generate_kitti_dataset_config(self):
        data_root = 'test/data/kitti'
        ann_file = 'labels.xlsx'
        classes = ['localizable', 'unlocalizable']
        # wait for pipline refactor
        pipeline = [
            dict(type='LoadPointsFromFile',
                 coord_type='LIDAR')
        ]

        modality = dict(use_lidar=True, use_camera=False)
        data_prefix = dict(pts='points')
        return data_root, ann_file, classes, data_prefix, pipeline, modality


    def test_getitem(self):
        DefaultScope.get_instance('test_simple_encoder', scope_name='mmdet3d')
        np.random.seed(0)
        data_root, ann_file, classes, data_prefix, \
            pipeline, modality, = self._generate_kitti_dataset_config()

        kitti_dataset = LocDataset(
            data_root,
            ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            metainfo=dict(classes=classes),
            modality=modality)

        kitti_dataset.prepare_data(0)
        input_dict = kitti_dataset.get_data_info(0)

        # assert the keys in ann_info and the type
        assert 'targets' in input_dict

        # only one instance
        assert 'lidar_path' in input_dict