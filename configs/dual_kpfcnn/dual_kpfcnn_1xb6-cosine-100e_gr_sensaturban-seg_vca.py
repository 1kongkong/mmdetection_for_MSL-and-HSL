_base_ = [
    "../_base_/datasets/sensaturban-seg.py",
    "../_base_/models/dual_kpfcnn.py",
    "../_base_/schedules/seg-cosine-80e.py",
    "../_base_/default_runtime.py",
]

# dataset settings
class_names = (
    "Ground",
    "Vegetation",
    "Building",
    "Wall",
    "Bridge",
    "Parking",
    "Rail",
    "Traffic Road",
    "Street Furniture",
    "Car",
    "Footpath",
    "Bike",
    "Water",
)

num_points = 8192
block_size = 20
backend_args = None
first_voxel_size = 0.2

train_pipeline = [
    dict(
        type="LoadPointsFromDict",
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
        sample_type="VoxelSp",
        sample_voxel_size=first_voxel_size,
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
        kpconv_channels=(
            (16, 32, 32),
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512),
        ),
        k_neighbor=25,
        sample_method="grid+rand",
        query_method="knn",
        weight_norm=True,
        norm_cfg=dict(type="BN1d", momentum=0.02),
        act_cfg=dict(type="LeakyReLU", negative_slope=0.1),
    ),  # [rgb, normalized_xyz]
    neck=dict(
        type="VectorCrossAttentionNeck",
        input_dims=(32, 64, 128, 256, 512),
        share_planes=(2, 2, 4, 8, 8),
    ),
    decode_head=dict(
        fp_channels=((1536, 512), (768, 256), (384, 128), (192, 128)),
        num_classes=len(class_names),
        ignore_index=len(class_names),
        loss_decode=dict(class_weight=None),
        stack=False,
        norm_cfg=dict(type="BN1d", momentum=0.02),
        act_cfg=dict(type="LeakyReLU", negative_slope=0.1),
    ),  # Titan doesn't use class_weight
    train_cfg=dict(stack=False, voxel_size=first_voxel_size),
    test_cfg=dict(
        num_points=num_points,
        block_size=block_size,
        mode="grid_slide",
        sample_rate=0.8,
        use_normalized_coord=True,
        batch_size=4,
    ),
)

# data settings
# train_dataloader = dict(batch_size=16)
train_dataloader = dict(
    batch_size=6,
    num_workers=8,
    persistent_workers=True,
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

# custom_hooks = [dict(type="EmptyCacheHook", after_iter=True)]

# 混合精度训练
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.001, betas=(0.95, 0.99)),
    clip_grad=None,
)
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=120,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=120,
    )
]

# PointNet2-MSG needs longer training time than PointNet2-SSG
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=2)
