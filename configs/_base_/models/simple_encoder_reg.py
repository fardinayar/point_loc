# model settings


model = dict(
    type='SimpleEncoder',
    head=dict(
        type='LinearRegressionHead',
        in_channels=512,
        loss=dict(type='L1Loss'),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



lr = 0.0002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.05),
    accumulative_counts=4
)
