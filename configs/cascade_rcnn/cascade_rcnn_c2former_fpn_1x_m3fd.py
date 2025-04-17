
_base_ = [
    '../_base_/datasets/m3fd.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py', '../_base_/models/cascade_rcnn_r50_fpn.py'
]

model = dict(
    type='Two_Stream_CascadeRCNN',
    backbone=dict(
        type='C2FormerResNet',
        fmap_size=(128, 160),
        dims_in=[256, 512, 1024, 2048],
        dims_out=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        cca_strides=[3, 3, 3, 3],
        groups=[1, 2, 3, 6],
        offset_range_factor=[2, 2, 2, 2],
        no_offs=[False, False, False, False],
        attn_drop_rate=0.0,
        drop_rate=0.0,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None,
        pretrained='pretrain_weights/resnet50-2stream.pth'
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=6),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=6),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=6)]))


dataset_type = 'M3FDDataset'
data_root = '/home/zp/Data/M3FD/VOC2012/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(600, 400), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PairedImageDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_tir', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'img_tir']),
            dict(type='Collect', keys=['img', 'img_tir']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                # data_root + 'VOC2007/ImageSets/Main/train.txt',
                data_root + 'ImageSets/Main/train.txt'
            ],
            img_prefix=[data_root],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')


# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# load_from='checkpoints/epoch_12.pth'