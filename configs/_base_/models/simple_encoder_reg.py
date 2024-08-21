# model settings

model = dict(
    type='SimpleEncoder',
    head=dict(
        type='MLPRegressionHead',
        num_outputs=21,
        hidden_channels=[512,512],
        in_channels=512,
        loss=dict(type='Hardshrink', lambd=0.001),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    accumulative_counts=10
    
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=200)
]
