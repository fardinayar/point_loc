
data_root_train = 'data/kitti/train'
data_root_test = 'data/kitti/test'

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        backend_args=None),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='PackMultiTaskInputs',
         task_handlers={key: dict(type='PackInputs',
                                  keys=['points']) for key in 'xyzabc'},
         multi_task_fields=['gt_label'])

]

val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        backend_args=None),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='PackMultiTaskInputs',
         task_handlers={key: dict(type='PackInputs',
                                  keys=['points']) for key in ['x','y','z','a','b','c']},
         multi_task_fields=['gt_label'])

]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='LocDataset',
        data_root=data_root_train,
        ann_file='labels.xlsx',
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
        type='LocDataset',
        data_root=data_root_test,
        ann_file='labels.xlsx',
        pipeline=val_pipeline,
        metainfo=dict(classes=['localizable', 'unlocalizable']),
        modality=dict(use_lidar=True, use_camera=False),
        backend_args=None))

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
