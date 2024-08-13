_base_ = [
    '../_base_/models/simple_encoder_reg.py', '../_base_/datasets/kitti_covariance.py'
]

default_scope = 'point_loc'


grid_shape = [480, 360, 16]

model = dict(#init_cfg=dict(type='Pretrained',#checkpoint='checkpoints/cylinder3d_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_144950-372cdf69.pth'),
    head=dict(
        type='LinearRegressionHead',
        num_classes=3,
        in_channels=512,
    ),
    data_preprocessor=dict(
        type='PointLocDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range= [-51.2, -51.2, -3.0, 51.2, 51.2, 1.0],
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

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001),
    paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=1),
    }),
    )

train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)
