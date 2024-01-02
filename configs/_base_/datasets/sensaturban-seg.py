# For SensatUrban seg
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
metainfo = dict(classes=class_names)
dataset_type = "SensatUrbanSegDataset"
data_root = "data/SensatUrban"
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts="points", pts_semantic_mask="semantic_mask")

backend_args = None

num_points = 8192
block_size = 30
# train_area = list(range(1,33))
all_paths = [
    "birmingham_block_0",
    "birmingham_block_1",
    "birmingham_block_10",
    "birmingham_block_11",
    "birmingham_block_12",
    "birmingham_block_13",
    "birmingham_block_2",
    "birmingham_block_3",
    "birmingham_block_4",
    "birmingham_block_5",
    "birmingham_block_6",
    "birmingham_block_7",
    "birmingham_block_8",
    "birmingham_block_9",
    "cambridge_block_10",
    "cambridge_block_12",
    "cambridge_block_13",
    "cambridge_block_14",
    "cambridge_block_15",
    "cambridge_block_16",
    "cambridge_block_17",
    "cambridge_block_18",
    "cambridge_block_19",
    "cambridge_block_2",
    "cambridge_block_20",
    "cambridge_block_21",
    "cambridge_block_22",
    "cambridge_block_23",
    "cambridge_block_25",
    "cambridge_block_26",
    "cambridge_block_27",
    "cambridge_block_28",
    "cambridge_block_3",
    "cambridge_block_32",
    "cambridge_block_33",
    "cambridge_block_34",
    "cambridge_block_4",
    "cambridge_block_6",
    "cambridge_block_7",
    "cambridge_block_8",
    "cambridge_block_9",
]
all_area = [path.split("/")[-1].split(".")[0] for path in all_paths]
test_area = [
    "birmingham_block_2",
    "birmingham_block_8",
    "cambridge_block_15",
    "cambridge_block_16",
    "cambridge_block_22",
    "cambridge_block_27",
]
val_area = [
    "birmingham_block_1",
    "birmingham_block_5",
    "cambridge_block_10",
    "cambridge_block_7",
]
train_area = [
    area for area in all_area if (area not in test_area) and (area not in val_area)
]

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
test_pipeline = [
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
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# we need to load gt seg_mask!
eval_pipeline = [
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
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

tta_pipeline = [
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
    # dict(type="NormalizePointsColor", color_mean=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(
                    type="RandomFlip3D",
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=0.0,
                )
            ],
            [dict(type="Pack3DDetInputs", keys=["points"])],
        ],
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[f"sensaturban_infos_{i}.pkl" for i in train_area],
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=[f"seg_info/{i}_resampled_scene_idxs.npy" for i in train_area],
        test_mode=False,
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[f"sensaturban_infos_{i}.pkl" for i in test_area],
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=[f"seg_info/{i}_resampled_scene_idxs.npy" for i in test_area],
        test_mode=True,
        backend_args=backend_args,
    ),
)
val_dataloader = test_dataloader

val_evaluator = dict(type="SegMetric")
test_evaluator = val_evaluator

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(type="Visualizer", vis_backends=vis_backends, name="visualizer")

tta_model = dict(type="Seg3DTTAModel")
