import copy
import os.path as osp
import unittest

import mmcv
import numpy as np
import torch
from PIL import Image

from point_loc.registry import TRANSFORMS
from mmpretrain.structures import DataSample


class TestPackInputs(unittest.TestCase):

    def test_transform(self):
        data = dict()
        
        points = torch.rand([1000, 3])
        data['points'] = points
        
        data['gt_label'] = 1
        '''{'x': 1,
                            'y': 1,
                            'z': 1,
                            'a': 0,
                            'b': 1,
                            'c': 0}'''
        
        cfg = dict(type='PackInputs', keys=['points'])
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        inputs = results['inputs']
        self.assertIn('points', inputs)
        self.assertIsInstance(inputs['points'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], DataSample)
        self.assertIsInstance(results['data_samples'].gt_label, torch.Tensor)
        
        
class TestPackMultiTaskInputs(unittest.TestCase):

    def test_transform(self):
        data = dict()
        
        points = torch.rand([1000, 3])
        data['points'] = points
        
        data['gt_label'] = {'x': 1,
                            'y': 1,
                            'z': 1,
                            'a': 0,
                            'b': 1,
                            'c': 0}
        
        cfg = dict(type='PackMultiTaskInputs', task_handlers={key: dict(type='PackInputs', keys=['points']) for key in ['x','y','z','a','b','c']}, multi_task_fields=['gt_label'])
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        inputs = results['inputs']
        self.assertIn('points', inputs)
        self.assertIsInstance(inputs['points'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], DataSample)
        self.assertIsInstance(results['data_samples'].x.gt_label, torch.Tensor)