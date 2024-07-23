
data_root = 'data/kitti_covariance_dataset/dataset'
ann_file = 'Covariance_train.txt'

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=3,
        backend_args=None),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.0001, 0.0001],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
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
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.0001, 0.0001],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='PackInputs',
        keys=['points'])

]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CovarianceLocDataset',
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        metainfo=dict(classes=['localizable', 'unlocalizable']),
        modality=dict(use_lidar=True, use_camera=False),
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CovarianceLocDataset',
        data_root=data_root,
        ann_file=ann_file,
        pipeline=val_pipeline,
        metainfo=dict(classes=['localizable', 'unlocalizable']),
        modality=dict(use_lidar=True, use_camera=False),
        backend_args=None))

train_cfg = dict(by_epoch=False, max_iters=100, val_interval=1)
