# model settings

model = dict(
    type='SimpleEncoder',
    head=dict(
        type='MLPRegressionHead',
        in_channels=512,
        hidden_channels=[256],
        num_outputs=21,
        num_shared_layers=0,
        loss=dict(type='KLDivergenceLoss'),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



optim_wrapper = dict(
    type='OptimWrapper',
    accumulative_counts=40,
    optimizer=dict(type='AdamW', lr=0.00001, weight_decay=0.01),    
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=1e-8,
        by_epoch=True,
        begin=0,
        end=200)
]
