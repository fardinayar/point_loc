# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_local_path
from mmdet3d.registry import DATASETS
import pandas as pd
import os


@DATASETS.register_module()
class LocDataset(BaseDataset):
    """Base Class for Localizabilty Estimation

    This is the base dataset of ScanNet, S3DIS and SemanticKITTI dataset.

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='points',
                 img='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used
            as input, it usually has following keys:

                - use_camera: bool
                - use_lidar: bool
            Defaults to dict(use_lidar=True, use_camera=False).
        scene_idxs (np.ndarray or str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        serialize_data (bool): Whether to hold memory using serialized objects,
            when enabled, data loader workers can use shared RAM from master
            process instead of making a copy.
            Defaults to False.
        load_eval_anns (bool): Whether to load annotations in test_mode,
            the annotation will be save in `eval_ann_infos`, which can be used
            in Evaluator. Defaults to True.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    METAINFO = {
        'classes': ['localizable', 'unlocalizable'],
    }
    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points',
                     img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 serialize_data: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        self.backend_args = backend_args
        self.modality = modality
        self.load_eval_anns = load_eval_anns

        self.label2cat = {
                i: cat_name
                for i, cat_name in enumerate(metainfo['classes'])
        }

        metainfo['label2cat'] = self.label2cat


        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=serialize_data,
            **kwargs)

        if not kwargs.get('lazy_init', False):
            self.scene_idxs = self.get_scene_idxs(scene_idxs)
            self.data_list = [self.data_list[i] for i in self.scene_idxs]

            # set group flag for the sampler
            if not self.test_mode:
                self._set_group_flag()


    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path.
        
        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])
            if 'num_pts_feats' in info['lidar_points']:
                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    img_info['img_path'] = osp.join(
                        self.data_prefix.get('img', ''), img_info['img_path'])


        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = dict()

        return info

    def prepare_data(self, idx: int) -> dict:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            dict: Results passed through ``self.pipeline``.
        """
        if not self.test_mode:
            data_info = self.get_data_info(idx)
            # Pass the dataset to the pipeline during training to support mixed
            # data augmentation, such as polarmix and lasermix.
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)

    def get_scene_idxs(self, scene_idxs: Union[None, str,
                                               np.ndarray]) -> np.ndarray:
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        if self.test_mode:
            # when testing, we load one whole scene every time
            return np.arange(len(self)).astype(np.int32)

        # we may need to re-sample different scenes according to scene_idxs
        # this is necessary for indoor scene segmentation such as ScanNet
        if scene_idxs is None:
            scene_idxs = np.arange(len(self))
        if isinstance(scene_idxs, str):
            scene_idxs = osp.join(self.data_root, scene_idxs)
            with get_local_path(
                    scene_idxs, backend_args=self.backend_args) as local_path:
                scene_idxs = np.load(local_path)
        else:
            scene_idxs = np.array(scene_idxs)

        return scene_idxs.astype(np.int32)

    def _set_group_flag(self) -> None:
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        
        
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        
        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        
        # Read annotations
        annotations = pd.read_excel(self.ann_file)
        points_path = sorted(os.listdir(self.data_prefix['pts']))
        annotations['point_path'] = points_path
        # Add file name to last
        raw_data_list = annotations.to_numpy()


        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info({'targets': raw_data_info[:6],
                                              'lidar_points':{'lidar_path': raw_data_info[-1]}
                                              })
            assert isinstance(data_info, dict)
            data_list.append(data_info)

        return data_list
