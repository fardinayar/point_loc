import unittest
import torch
from mmengine import DefaultScope
from mmdet3d.registry import MODELS
from mmdet3d.testing import (get_detector_cfg,
                             setup_seed)
from mmengine.config import Config
from point_loc.testing.model_utils import create_sample_inputs
import logging as log

'''import sys 
sys.path.append('/home/fardin/point_localizability_estimation/point_loc') '''

class TestSimpleEncoder(unittest.TestCase):

    def test_simple_encoder_pointnet2(self):
        import point_loc.models
        assert hasattr(point_loc.models, 'SimpleEncoder')
        DefaultScope.get_instance('test_simple_encoder', scope_name='mmdet3d')
        setup_seed(0)
        simple_encoder_cfg = Config.fromfile('configs/simple_encoder/simple_encoder_pointnet2.py').model
        simple_encoder_cfg.head.num_outputs = 6
        model = MODELS.build(simple_encoder_cfg)
        packed_inputs = create_sample_inputs(
            num_classes=1)
        
        self.assertEqual(torch.cuda.is_available(), True)
        
        model = model.cuda()
        # test simple_test
        data = model.data_preprocessor(packed_inputs, True)
        torch.cuda.empty_cache()
        results = model.forward(**data, mode='predict')
        results = model.forward(**data, mode='tensor')
        self.assertEqual(results.shape, (1,6))

        losses = model.forward(**data, mode='loss')
        
        self.assertIsInstance(losses, dict)
            
        loss = sum([losses[k] for k in losses.keys()])
        loss.backward()
        
    def test_simple_encoder_cylinder3d(self):
        import point_loc.models
        assert hasattr(point_loc.models, 'SimpleEncoder')
        DefaultScope.get_instance('test_simple_encoder', scope_name='mmdet3d')
        setup_seed(0)
        simple_encoder_cfg = Config.fromfile('configs/simple_encoder/simple_encoder_cylinder3d.py').model
        simple_encoder_cfg.head.num_outputs = 6
        model = MODELS.build(simple_encoder_cfg)
        packed_inputs = create_sample_inputs(
            num_classes=1)
        
        self.assertEqual(torch.cuda.is_available(), True)
        
        model = model.cuda()
        # test simple_test
        data = model.data_preprocessor(packed_inputs, True)
        torch.cuda.empty_cache()
        results = model.forward(**data, mode='predict')
        results = model.forward(**data, mode='tensor')
        self.assertEqual(results.shape, (1,6))

        losses = model.forward(**data, mode='loss')
        
        self.assertIsInstance(losses, dict)
            
        loss = sum([losses[k] for k in losses.keys()])
        loss.backward()
        
        