# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.testing.model_utils import setup_seed
import numpy as np
import torch


def create_sample_inputs(seed=0,
                        with_points=True,
                        with_img=False,
                        img_size=10,
                        num_points=10,
                        points_feat_dim=4,
                        num_classes=1,
                        num_targets=6):
    
    assert num_classes == 1, "Number of classes should be 1. We only support binary classification and regression for now"
    setup_seed(seed)

    meta_info = dict()
    meta_info['depth2img'] = np.array(
        [[5.23289349e+02, 3.68831943e+02, 6.10469439e+01],
         [1.09560138e+02, 1.97404735e+02, -5.47377738e+02],
         [1.25930002e-02, 9.92229998e-01, -1.23769999e-01]])
    meta_info['lidar2img'] = np.array(
        [[5.23289349e+02, 3.68831943e+02, 6.10469439e+01],
         [1.09560138e+02, 1.97404735e+02, -5.47377738e+02],
         [1.25930002e-02, 9.92229998e-01, -1.23769999e-01]])

    inputs_dict = dict()

    if with_points:
        points = torch.rand([num_points, points_feat_dim])
        inputs_dict['points'] = [points]

    if with_img:
        if isinstance(img_size, tuple):
            img = torch.rand(3, img_size[0], img_size[1])
            meta_info['img_shape'] = img_size
            meta_info['ori_shape'] = img_size
        else:
            img = torch.rand(3, img_size, img_size)
            meta_info['img_shape'] = (img_size, img_size)
            meta_info['ori_shape'] = (img_size, img_size)
        meta_info['scale_factor'] = np.array([1., 1.])
        inputs_dict['img'] = [img]

    data_samples = torch.rand(num_targets)
    # TODO: correct data_samples
    return dict(inputs=inputs_dict, data_samples=[data_samples])
