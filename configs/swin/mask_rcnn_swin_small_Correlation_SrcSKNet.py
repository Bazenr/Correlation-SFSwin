_base_ = './mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py'

# model = dict(
#     backbone=dict(
#         depths=[2, 2, 18, 2],
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa


dataset_type = 'CocoDataset'  # suffix='bmp', in_chans2=3, mean, std =...[:6]
data_root = R"F:/Keras-Tf-Mask_rcnn_Jiantao/2022_9_10_PE_PP_PET/Band_73_80_207/coco_9Types_png/"
classes = ("Color_PP","Trans_PET","Color_PET",  # 标签已改
           "Trans_PP","White_PE","White_PET",
           "Trans_PE","White_PP","Color_PE")

load_from = R'H:\projects\mmdetr_rebuild\pretrain_model/' + \
              'correlation_swin_rebuild/pretrain_correlation_swin_rebuild.pth'

pretrained = R'H:\projects\mmdetr_rebuild\pretrain_model/' + \
              'swin_small_patch4_window7_224.pth'  # noqa

mean = ([123.675, 116.28, 103.53] * 76)[:6]
std  = ([58.395,  57.12,  57.375] * 76)[:6]

fpn_channels = [384, 768, 1536, 3072]  # [96, 192, 384, 768], [192, 384, 768, 1536]

model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetFusionDataPreprocessor',  # DetDataPreprocessor
        bgr_to_rgb=False,
        mean=mean,
        std=std),
    backbone=dict(
        _delete_=True,
        type='Corr_SrcSKNet_SwinTransformer',  # 
        in_chans1=3,  # 
        in_chans2=3,  # 
        embed_dims=96,
        # # 老版SKNet参数
        SK_channels = fpn_channels,
        M = 4,   # Split步骤的分支数, # 双主干时使用2
        #  WH = ?,  # 图像长宽
        G = 32,  # Split步骤卷积的分组数, 32
        r = 8,   # Fuse步骤FC层的缩减率, 16
        L = 32,  # 原文默认32
        # # 
        # 与之前的区别1
        depths=[2, 2, 6, 2],  # 6， 18
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
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)  # #############
        ),
    neck=dict(in_channels=fpn_channels),  # [96, 192, 384, 768]
    roi_head=dict(
        bbox_head=dict(num_classes=9),
        mask_head=dict(num_classes=9))
)

# 与之前的区别2
max_epochs = 26
train_cfg = dict(max_epochs=max_epochs)

# # Old learning rate
# param_scheduler = [
#     dict(  # start_factor越大, 起步学习率越高
#          type='LinearLR', start_factor=0.001 , by_epoch=True, begin=0, end=6),
#     dict(type='CosineAnnealingLR', by_epoch=True, T_max=5, begin=5, end=14, eta_min=2e-7),
#     dict(type='ExponentialLR', begin=14, end=21, gamma=0.4),  # gamma大则缓
#     dict(type='ConstantLR', begin=21, end=24, factor=0.3)
#         ]

# New learning rate
param_scheduler = [
    dict(  # start_factor越大, 起步学习率越高
         type='LinearLR', start_factor=0.001 , by_epoch=True, begin=0, end=6),
    dict(type='ExponentialLR', begin=5, end=27, gamma=0.69),  # gamma大则缓
        ]

# optimizer
optim_wrapper = dict(
    # 与之前的区别4
    type='AmpOptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,  # 0.0001
        # beta1 计算梯度的指数移动平均值的衰减率, 衰减率越大, 之前的梯度权重越大，模型越平稳
        # beta2 计算梯度平方的指数移动平均值的衰减率, 防止梯度的更新在某一方向上过于激进
        betas=(0.9, 0.999),  # (0.9, 0.999)
        # weight_decay 权重衰减是一种正则化技术，用于防止模型过拟合
        weight_decay=0.05))  # 0.05

backend_args = None

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadFusionDataFromFiles', suffix='bmp', if_fusion=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')) # 少加这部分导致loss为0

]
test_pipeline = [
    dict(type='LoadFusionDataFromFiles', suffix='bmp', if_fusion=True, backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]

# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
train_dataloader = dict(
    batch_size=1,
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

