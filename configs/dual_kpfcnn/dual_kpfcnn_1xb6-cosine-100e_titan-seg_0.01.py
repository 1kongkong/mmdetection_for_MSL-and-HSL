_base_ = [
    "../_base_/datasets/titan-seg.py",
    "../_base_/models/dual_kpfcnn.py",
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
        load_dim=6,  # 原数据的列数
        use_dim=[0, 1, 2, 3, 4, 5],  # 保留的列
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
        in_channels=6,
        # sample_nums=(2000, 500, 250, 125),
        weight_norm=True,
    ),  # [rgb, normalized_xyz]
    decode_head=dict(
        num_classes=len(class_names),
        ignore_index=len(class_names),
        loss_decode=dict(class_weight=None),
    ),  # Titan doesn't use class_weight
    test_cfg=dict(
        num_points=num_points,
        block_size=block_size,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=8,
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
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=0.01, weight_decay=0.001, betas=(0.95, 0.99)),
    clip_grad=None,
)
# optim_wrapper = dict(
#     type="OptimWrapper",
#     optimizer=dict(type="AdamW", lr=0.01, weight_decay=0.001, betas=(0.95, 0.99)),
#     clip_grad=None,
# )
param_scheduler = [
    dict(
        T_max=80,
        begin=0,
        by_epoch=True,
        end=80,
        eta_min=1e-04,
        type='CosineAnnealingLR'),
]

# PointNet2-MSG needs longer training time than PointNet2-SSG
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=2)
