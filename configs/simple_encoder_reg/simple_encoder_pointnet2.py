_base_ = [
    '../_base_/models/simple_encoder_reg.py', '../_base_/datasets/kitti_covariance.py'
]

default_scope = 'point_loc'
model = dict(
    head=dict(
        type='MLPRegressionHead',
        hidden_channels=[512, 128, 32],
        num_outputs=21,
        num_shared_layers=1,
        in_channels=256,
    ),
    data_preprocessor=dict(type='PointLocDataPreprocessor'),
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=3,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    neck=dict(type='PoolingNeck',
              feature_name='sa_features'),
)
