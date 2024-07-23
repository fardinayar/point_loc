# model settings


model = dict(
    type='SimpleEncoder',
    head=dict(
        type='LinearRegressionHead',
        num_classes=21,
        in_channels=512,
        loss=dict(type='MSELoss'),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='MeanAbsoluteError')


lr = 0.002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
    accumulative_counts=4
)
