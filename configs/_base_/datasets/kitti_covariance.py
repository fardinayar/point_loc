
data_root = 'data/kitti_covariance_dataset/dataset'

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=3,
        backend_args=None),
    dict(
        type='RandomJitterPoints'
    ),
    dict(
        type='GlobalRotScaleTransWithCov',
        rot_range=[-0.78539816*4, 0.78539816*4],
        scale_ratio_range=[0.99, 1.01],
        translation_std=[5, 5, 5],
    ),
    dict(type='PointSample', num_points=30000),
    dict(type='PackInputs',
        keys=['points'])

]

val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=3,
        backend_args=None),
    dict(type='PointSample', num_points=30000),
    dict(type='PackInputs',
        keys=['points'])

]

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CovarianceLocDataset',
        data_root=data_root,
        ann_file='Covariance_train.txt',
        pipeline=train_pipeline,
        metainfo=dict(classes=['localizable', 'unlocalizable']),
        modality=dict(use_lidar=True, use_camera=False),
        backend_args=None))

val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CovarianceLocDataset',
        data_root=data_root,
        ann_file='Covariance_validation.txt',
        pipeline=val_pipeline,
        metainfo=dict(classes=['localizable', 'unlocalizable']),
        modality=dict(use_lidar=True, use_camera=False),
        backend_args=None))

test_dataloader = val_dataloader
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=2)

val_cfg = dict(type='ValLoop')
test_cfg = val_cfg
val_evaluator = dict(type='Evaluator',
                     metrics=[dict(type='MeanAbsoluteError'), dict(type='RelativeDelta'), dict(type='KLDivergence')])
test_evaluator = val_evaluator





