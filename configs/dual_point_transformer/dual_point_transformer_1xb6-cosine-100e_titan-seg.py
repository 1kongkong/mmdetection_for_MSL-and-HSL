_base_ = [
    "../_base_/datasets/titan-idw-seg.py",
    "../_base_/models/dual_point_transformer.py",
    "../_base_/schedules/seg-cosine-80e.py",
    "../_base_/default_runtime.py",
]

# dataset settings
class_names = (
    "Impervious Ground",
    "Grass",
    "Building",
    "Tree",
    "Car",
    "Power Line",
    "Bare land",
)

num_points = 8192
block_size = 30
backend_args = None
train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=13,  # 原数据的列数
        use_dim=[0, 1, 2, 6, 7, 8],  # 保留的列
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args,
    ),
    dict(type="PointSegClassMapping"),
    dict(
        type="IndoorPatchPointSample",
        num_points=num_points,
        block_size=block_size,
        ignore_index=len(class_names),
        use_normalized_coord=True,
        enlarge_size=0.2,
        min_unique_num=None,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.14159264, 3.14159264],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "pts_semantic_mask"]),
]

# model settings
model = dict(
    backbone=dict(
        in_channels_spa=3,
        in_channels_spe=3,
        num_points=(8192, 2048, 512, 128, 32),
        num_samples=(8, 16, 16, 16, 16),
    ),  # [rgb, normalized_xyz]
    neck=dict(
        type="VectorCrossAttentionNeck",
        input_dims=(16, 32, 64, 128, 256),
        share_planes=(2, 2, 4, 8, 8),
    ),
    decode_head=dict(
        num_classes=len(class_names),
        ignore_index=len(class_names),
        loss_decode=dict(class_weight=None),
    ),  # Titan doesn't use class_weight
    test_cfg=dict(
        num_points=num_points,
        block_size=block_size,
        mode="slide",
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=16,
    ),
)

# data settings
# train_dataloader = dict(batch_size=16)
train_dataloader = dict(
    batch_size=8,
    num_workers=12,
    dataset=dict(pipeline=train_pipeline),
)

# runtime settings
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=2,
        by_epoch=True,
        save_best=["acc", "miou"],
        rule="greater",
    )
)

# 混合精度训练
# optim_wrapper = dict(
#     type="AmpOptimWrapper",
#     loss_scale="dynamic",
#     optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.001, betas=(0.95, 0.99)),
#     clip_grad=None,
# )

# PointNet2-MSG needs longer training time than PointNet2-SSG
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=2)
