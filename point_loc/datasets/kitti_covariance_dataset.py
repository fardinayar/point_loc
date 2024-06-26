# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
#from mmengine.dataset import BaseDataset
#from mmengine.fileio import get_local_path
#from mmdet3d.registry import DATASETS
import pandas as pd
import os



# Function to read the text file and create matrices
def read_text_file(filename):
    matrices = []
    points = []

    with open(filename, 'r') as file:
        for line in file:
            # Split the line into components
            components = line.strip().split()
            
            # First component is the point name
            point_name = components[0]
            points.append(point_name)
            
            # The rest are the elements of the matrix
            matrix_elements = list(map(float, components[1:]))
            
            # Reshape the elements into a 6x6 matrix
            matrix = np.array(matrix_elements).reshape(6, 6)
            matrices.append(matrix)
    
    return points, matrices

def give_indices(matrix):
    # Extract upper triangular part of the matrix and their indices
    indices = np.triu_indices(matrix.shape[0])
    return indices



#@DATASETS.register_module()
class LocDataset():
    METAINFO = {
        'classes': ['localizable', 'unlocalizable'],
    }

    def __init__(self,
                 ann_file: str,
                 data_root: Optional[str] = None,
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
        
        if metainfo is None:
            metainfo = {'classes': ['localizable', 'unlocalizable']}

        self.label2cat = {
                i: cat_name
                for i, cat_name in enumerate(metainfo['classes'])
        }

        metainfo['label2cat'] = self.label2cat
        self.backend_args = backend_args
        self.modality = modality
        self.load_eval_anns = load_eval_anns
        self.ann_file = ann_file
        self.data_root = data_root
        self.data_prefix = data_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.serialize_data = serialize_data

        # Initialize data_list as an empty list
        self.data_list = []

        if not kwargs.get('lazy_init', False):
            self.scene_idxs = self.get_scene_idxs(scene_idxs)
            self.data_list = [self.data_list[i] for i in self.scene_idxs]

            if not self.test_mode:
                self._set_group_flag()

    def get_scene_idxs(self, scene_idxs: Union[None, str, np.ndarray]) -> np.ndarray:
        """Compute scene indexes for data sampling."""
        if self.test_mode:
            return np.arange(len(self.data_list)).astype(np.int32)

        if scene_idxs is None:
            scene_idxs = np.arange(len(self.data_list))
        elif isinstance(scene_idxs, str):
            scene_idxs = osp.join(self.data_root, scene_idxs)
            scene_idxs = np.load(scene_idxs)
        else:
            scene_idxs = np.array(scene_idxs)

        return scene_idxs.astype(np.int32)

    def make_path_files(self, file_name):
        sequence_dir = os.path.join(self.data_root, "sequences")
        
        # Extracting file number parts
        #file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_parts = file_name.split('_')
        # Extracting components
        sequence_number = file_parts[0]  
        point_cloud_number = file_parts[1] 
        current_file_path = os.path.join(sequence_dir+ sequence_number +'/'+'velodyne/'+f"{point_cloud_number}.bin")
        return current_file_path

    def _set_group_flag(self) -> None:
        """Set flag according to image aspect ratio."""
        if hasattr(self, 'data_list'):  # Check if data_list is defined
            self.flag = np.zeros(len(self.data_list), dtype=np.uint8)
        else:
            raise RuntimeError("data_list is not initialized properly.")
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
    def load_points(self):
        points, matrices = read_text_file(self.ann_file)
        return points    
    
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        
        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        
        # Read annotations
        #annotations = pd.read_excel(self.ann_file)
        #points_path = sorted(os.listdir(self.data_prefix['pts']))
        #annotations['point_path'] = points_path
        # Add file name to last
        #raw_data_list = annotations.to_numpy()
        points, matrices = read_text_file(self.ann_file)
        point_of_path = [self.make_path_files(point) for point in points]
        #point_of_path=[]
        #for point in points:
            #points_path = self.make_path_files(point)
            #point_of_path.append(points_path)


        # load and parse data_infos.
        data_list = []
        for i in range(len(matrices)):
            indices = give_indices(matrices[i])
            upper_triangular = matrices[i][indices]
            # Create keys in the 'a00' format
            keys = [f'a{indices[0][j]}{indices[1][j]}' for j in range(len(upper_triangular))]
            # Create gt_label dictionary
            gt_label = {keys[k]: upper_triangular[k] for k in range(len(keys))}

            # parse raw data information to target format
            data_info = self.parse_data_info({'gt_label': gt_label,
                                              'lidar_points':{'lidar_path': point_of_path[i]}
                                              })
            assert isinstance(data_info, dict)
            data_list.append(data_info)

        return data_list
    
    
