_base_ = [
    '../_base_/datasets/s3dis-seg.py', '../_base_/models/kpfcnn.py',
    '../_base_/schedules/seg-cosine-50e.py', '../_base_/default_runtime.py'
]

# dataset settings
class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
num_points = 4096
train_pipeline = [
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.0,
        ignore_index=len(class_names),
        use_normalized_coord=True,
        enlarge_size=0.2,
        min_unique_num=None),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
# model settings
model = dict(
    backbone=dict(
        in_channels=6,
        sample_nums=(2048, 1024, 512, 256),
        weight_norm=True,
    ),  # [rgb, normalized_xyz]
    decode_head=dict(
        num_classes=13,
        ignore_index=13,
        loss_decode=dict(class_weight=None)
    ),  # S3DIS doesn't use class_weight
    test_cfg=dict(
        num_points=num_points,
        block_size=1.0,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=24))

# data settings
# train_dataloader = dict(batch_size=16)
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
)

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))

# PointNet2-MSG needs longer training time than PointNet2-SSG
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)
