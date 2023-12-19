# model settings
model = dict(
    type="EncoderDecoder3D",
    data_preprocessor=dict(type="Det3DDataPreprocessor"),
    backbone=dict(
        type="RandLANetBackbone",
        num_points=(8192, 2048, 512, 128, 64),
        in_channels=6,  # [xyz, rgb], should be modified with dataset
        num_samples=(16, 16, 16, 16, 16),
        enc_channels=(
            (8, 16),
            (32, 64),
            (128, 128),
            (256, 256),
        ),
    ),
    decode_head=dict(
        type="RandLANetHead",
        dec_channels=(
            (1024, 256),
            (512, 128),
            (256, 32),
            (64, 32),
        ),
        channels=32,
        dropout_ratio=0.5,
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="LeakyReLU", negative_slope=0.1),
        loss_decode=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            class_weight=None,  # should be modified with dataset
            loss_weight=1.0,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="slide"),
)
