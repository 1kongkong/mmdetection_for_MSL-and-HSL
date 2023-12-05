# model settings
model = dict(
    type="EncoderDecoder3D",
    data_preprocessor=dict(type="Det3DDataPreprocessor"),
    backbone=dict(
        type="PointTransformerBackbone",
        in_channels=6,  # [xyz, rgb], should be modified with dataset
        num_points=(8192, 2048, 512, 128, 32),
        num_samples=(8, 16, 16, 16, 16),
        enc_channels=(
            (32, 32),
            (64, 64, 64),
            (128, 128, 128, 128),
            (256, 256, 256, 256, 256, 256),
            (512, 512, 512),
        ),
    ),
    decode_head=dict(
        type="PointTransformerHead",
        channels=128,
        num_samples=(16, 16, 16, 16, 8),
        dec_channels=(
            (512, 512),
            (256, 256),
            (128, 128),
            (64, 64),
            (32, 32),
        ),
        dropout_ratio=0.0,
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="ReLU"),
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
