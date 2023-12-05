# model settings
model = dict(
    type="EncoderDecoder3D",
    data_preprocessor=dict(type="Det3DDataPreprocessor"),
    backbone=dict(
        type="Dual_KPFCNNBackbone",
        num_point=8192,
        in_channels=6,  # [xyz, rgb], should be modified with dataset
        kernel_size=15,
        k_neighbor=20,
        sample_nums=(2048, 512, 128, 64),
        kpconv_channels=(
            (16, 32, 32),
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512),
        ),
        weight_norm=False,
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="LeakyReLU", negative_slope=0.1),
    ),
    decode_head=dict(
        type="KPFCNNHead",
        fp_channels=((1536, 512), (768, 256), (384, 128), (192, 128)),
        channels=128,
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
