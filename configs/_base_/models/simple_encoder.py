# model settings

model = dict(
    type='SimpleEncoder',
    head=dict(
        type='mmpretrain.MultiTaskHead',
        task_heads ={'x': dict(type='LinearClsHead'),
                     'y': dict(type='LinearClsHead'),
                     'z': dict(type='LinearClsHead'),
                     'a': dict(type='LinearClsHead'),
                     'b': dict(type='LinearClsHead'),
                     'c': dict(type='LinearClsHead')},
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='mmpretrain.MultiTasksMetric',
                     task_metrics={key: [dict(type='mmpretrain.Accuracy')] for key in 'xyzabc'})


lr = 0.002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
)
