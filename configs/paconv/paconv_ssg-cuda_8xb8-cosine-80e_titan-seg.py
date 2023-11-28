_base_ = [
    "../_base_/datasets/titan-seg.py",
    "../_base_/models/paconv_ssg-cuda.py",
    "../_base_/schedules/seg-cosine-50e.py",
    "../_base_/default_runtime.py",
]

# dataset settings
class_names = ("Impervious Ground", "Grass", "Building", "Tree", "Car", "Power Line", "Bare land")

num_points = 50000
block_size = 75
backend_args = None
train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
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
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type="Pack3DDetInputs", keys=["points", "pts_semantic_mask"]),
]

# model settings
model = dict(
    backbone=dict(
        in_channels=9,
        num_points=(12500, 3125, 780, 195),
    ),
    decode_head=dict(
        num_classes=7, ignore_index=7, loss_decode=dict(class_weight=None)
    ),  # S3DIS doesn't use class_weight
    test_cfg=dict(num_points=50000, block_size=75, sample_rate=0.5, use_normalized_coord=True, batch_size=1),
)

train_dataloader = dict(batch_size=4, num_workers=8, dataset=dict(pipeline=train_pipeline))

# runtime settings
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=2))

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)
