# 原mask-rcnn_r50-dconv-c3-c5_fpn_amp-1x_coco
_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'  # base: mask-rcnn_r50_fpn.py
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

# # MMEngine support the following two ways, users can choose
# # according to convenience
# optim_wrapper = dict(type='AmpOptimWrapper')
# _base_.optim_wrapper.type = 'AmpOptimWrapper'

dataset_type = 'CocoDataset'  # suffix='bmp', in_chans2=3, mean, std =...[:6]
data_root = R"F:/Keras-Tf-Mask_rcnn_Jiantao/2022_9_10_PE_PP_PET/Band_73_80_207/coco_9Types_png/"
classes = ("Color_PP","Trans_PET","Color_PET",  # 标签已改
           "Trans_PP","White_PE", "White_PET",
           "Trans_PE","White_PP", "Color_PE")

# # 原论文的代码的预训练权重
# load_from = 'D:/Keras-Tf-Mask_rcnn_Jiantao/mmdetection/pretrain_model/concat_pretrain_model.pth'
# # 新代码预训练权重
# pretrained = R'H:\projects\mmdetr\pretrain_model/double_resnet/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203-double_resnet.pth'
# # 新代码预训练权重
load_from = R'H:\projects\mmdetr_rebuild\pretrain_model/double_resnet_101_dcn/mask_rcnn_double_resnet101.pth'

mean = ([123.675, 116.28, 103.53] * 76)[:6]
std  = ([58.395,  57.12,  57.375] * 76)[:6]

model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetFusionDataPreprocessor',  # DetDataPreprocessor
        bgr_to_rgb=False,
        mean=mean,
        std=std),
    backbone=dict(
        type='DoubleResNet_src',  # DoubleResNet, DoubleResNet_src, DoubleResNet_SKNet_divide
        in_chans1=3,
        in_chans2=3,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048, 4096],  # [256, 512, 1024, 2048]
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=9,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=9,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'D:/Keras-Tf-Mask_rcnn_Jiantao/mmdetection/pretrain_model/ResNet50_concat.pth'
            # 'H:\projects\mmdetr\pretrain_model/double_resnet/ResNet50_concat.pth'
            )
            )

max_epochs = 26  # 24, 36, 26
train_cfg = dict(max_epochs=max_epochs)

'''Swin融合设计'''
# learning rate
param_scheduler = [
    dict(# start_factor越大, 起步学习率越高
         type='LinearLR', start_factor=0.001 , by_epoch=True, begin=0, end=6),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=5, begin=5, end=14, eta_min=2e-7),
    dict(type='ExponentialLR', begin=14, end=21, gamma=0.4),  # gamma大则缓
    dict(type='ConstantLR', begin=21, end=24, factor=0.3)
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
        lr=0.0001,  # 0.0001
        # beta1 计算梯度的指数移动平均值的衰减率, 衰减率越大, 之前的梯度权重越大，模型越平稳
        # beta2 计算梯度平方的指数移动平均值的衰减率, 防止梯度的更新在某一方向上过于激进
        betas=(0.9, 0.999),  # (0.9, 0.999)
        # weight_decay 权重衰减是一种正则化技术，用于防止模型过拟合
        weight_decay=0.05))  # 0.05

'''Swin原设计'''
# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
#         end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[10, 14],
#         gamma=0.1)
# ]
# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }),
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=0.00005,
#         betas=(0.9, 0.999),
#         weight_decay=0.05))

'''mmdetection原设计'''
# # 原版双ResNet的参数
# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# # optimizer
# optim_wrapper = dict(
#     # 与之前的区别4
#     type='AmpOptimWrapper',
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }),
#     optimizer=dict(
#         _delete_=True,
#         type='SGD',
#         lr=0.02,
#         momentum=0.9,
#         weight_decay=0.0001))  # 0.05

# auto_scale_lr = dict(enable=True, base_batch_size=16)

# # mmpretrain 学习率及优化器参考
# optim_wrapper = dict(
#     optimizer=dict(lr=0.003, type='Lamb', weight_decay=0.01),  # 这个在det没定义
#     paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))


backend_args = None
# from torch.optim.adamw import AdamW
# from torch.optim.Lamb import Lamb

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
    batch_size=4,
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
