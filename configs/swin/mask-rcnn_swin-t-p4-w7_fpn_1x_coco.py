_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'  # suffix='bmp', in_chans2=3, mean, std =...[:6]
data_root = R"F:/Keras-Tf-Mask_rcnn_Jiantao/2022_9_10_PE_PP_PET/Band_73_80_207/coco_9Types_png/"
classes = ("Color_PP","Trans_PET","Color_PET",  # 标签已改
           "Trans_PP","White_PE","White_PET",
           "Trans_PE","White_PP","Color_PE")

# dataset_type = 'CocoDataset'  # suffix='tif', in_chans2=224, mean, std =...[:227]
# data_root = R"F:\Keras-Tf-Mask_rcnn_Jiantao\2022_9_10_PE_PP_PET_FactoryCut_src\Coco_Train_val_copypaste_full/"
# classes = ("Trans_PET","Color_PE","Trans_PP",
#            "White_PET","White_PE","Color_PET",
#            "Trans_PE", "White_PP","Color_PP")

mean = ([123.675, 116.28, 103.53] * 76)[:6]
std  = ([58.395,  57.12,  57.375] * 76)[:6]

fpn_channels = [384, 768, 1536, 1536*2]  # [192, 384, 768, 1536]

pretrained = R'H:\projects\mmdetr\pretrain_model/correlation_backbone/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_correlation_backbone.pth'

model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetFusionDataPreprocessor',  # DetDataPreprocessor
        bgr_to_rgb=False,
        mean=mean,
        std=std),
    backbone=dict(
        _delete_=True,
        type='Fus_Correlation_SwinT',  # SwinTransformer, SKNet_SwinTransformer, Fus_Correlation_SwinT
        # # 
        in_chans1=3,
        in_chans2=3,  # 3. 224
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=dict(in_channels=fpn_channels)  # [96, 192, 384, 768]
    # 
    )

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

backend_args = None

train_pipeline = [
    dict(type='LoadFusionDataFromFiles', suffix='bmp', if_fusion=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')]
test_pipeline = [
    dict(type='LoadFusionDataFromFiles', suffix='bmp', if_fusion=True, backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))]

train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False)
test_evaluator = val_evaluator
