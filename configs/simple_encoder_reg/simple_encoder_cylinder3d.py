_base_ = [
    '../_base_/models/simple_encoder_reg.py', '../_base_/datasets/kitti_covariance.py'
]

default_scope = 'point_loc'


grid_shape = [480, 360, 32]

model = dict(
    head=dict(
        type='MLPRegressionHead',
        hidden_channels=[512, 128, 32],
        num_outputs=21,
        num_shared_layers=1,
        in_channels=512,
        loss=dict(type='MSELoss'),
    ),
    data_preprocessor=dict(
        type='PointLocDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),

    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=5,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    
    backbone=dict(
        type='Cylinder3DBackbone',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    
    neck=dict(type='PoolingNeck'),
)
