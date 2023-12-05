_base_ = [
    "../_base_/datasets/titan-seg.py",
    "../_base_/models/kpfcnn.py",
    "../_base_/schedules/seg-cosine-200e.py",
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

num_points = 50000
block_size = 50
backend_args = None
first_voxel_size = 0.4

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
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=1e-10,
        flip_box3d=False,
    ),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(
        type="RandomJitterPoints",
    ),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type="Pack3DDetInputs", keys=["points", "pts_semantic_mask"]),
]
# model settings
model = dict(
    backbone=dict(
        in_channels=6,
        weight_norm=False,
        voxel_size=[0.8, 1.6, 3.2, 6.4],
        radius=[1.0, 2.0, 4.0, 8.0, 16.0],
        norm_cfg=dict(type="GN", num_groups=2),
    ),  # [rgb, normalized_xyz]
    decode_head=dict(
        num_classes=7,
        ignore_index=7,
        loss_decode=dict(class_weight=None),
        stack=False,
        norm_cfg=dict(type="GN", num_groups=2),
    ),  # Titan doesn't use class_weight
    train_cfg=dict(stack=False, voxel_size=first_voxel_size),
    test_cfg=dict(
        num_points=num_points,
        block_size=block_size,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=6,
    ),
)

# data settings
# train_dataloader = dict(batch_size=16)
train_dataloader = dict(
    batch_size=6,
    num_workers=6,
    dataset=dict(pipeline=train_pipeline),
)

# runtime settings
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        by_epoch=True,
        save_best=["acc", "miou"],
        rule="greater",
    )
)

# 混合精度训练
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="Adam", lr=0.001, weight_decay=0.001),
    clip_grad=None,
)

# PointNet2-MSG needs longer training time than PointNet2-SSG
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=5)
