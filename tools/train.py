# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


'''参数文件mask-rcnn_swin-t-p4bu-w7_fpn_1x_coco.py位置
    H:\projects\mmdetr_rebuild\configs\swin
    H:\projects\mmdetr_rebuild\configs\_base_\datasets\coco_instance.py
    H:\projects\mmdetr_rebuild\configs\_base_\schedules\schedule_1x.py
''' 
work_path = R"H:\projects\mmdetr_rebuild\configs\swin/"

out_fold = "correlation_swin_SrcSKNet_6h_4"  #
# double_swin_rebuild, double_swin_rebuild_newpre, _2, _3, _4, _5, _6，_7 新设计预训练权重  # 在swin内部重建的双主干尝试 (mask_rcnn_swin_small)
# double_swin_SrcSKNet, _2, _4, _5, _6                                          原版SKNet (mask_rcnn_swin_small_SrcSKNet)
# double_swin_DivideSKNet_M=3问题结果
# double_swin_DivideSKNet, _2, _3, double_swin_DivideSKNet_decrease, _2           新版分离SKNet (mask_rcnn_swin_small_DivideSKNet) 包括递增递减
# correlation_swin_not_full_pre, correlation_swin, _2, _3, _5, _6_newLr           四主干互注意力算法 (mask_rcnn_swin_small_Correlation)
# correlation_swin_DivideSKNet, _2, _3                                            四主干+新版分离SKNet算法 (mask_rcnn_swin_small_Correlation_DivideSKNet)
# correlation_swin_SrcSKNet, _2, _4, _5_newLr, _6_newLr                           四主干+原版SKNet算法 (mask_rcnn_swin_small_Correlation_SrcSKNet)
# correlation_swin_SF_SKNet, _2, _3                                               先SFNet后SKNet (mask_rcnn_swin_small_Correlation_SF_SKNet)
# ##接上 correlation_swin_DivideSKNet_decrease, _2, _3_newLr, _4, _5, _6_newLr, _8, _9    (mask_rcnn_swin_small_Correlation_DivideSKNet)
# ##接上 correlation_swin_DivideSKNet_increase, _2, _3_newLr                        (mask_rcnn_swin_small_Correlation_DivideSKNet)
# ##接上 correlation_swin_DivideSKNet_ZZZ2222, correlation_swin_DivideSKNet_2222_2  (mask_rcnn_swin_small_Correlation_DivideSKNet)
# double_resnet_101_dcnv2, _2, _3, _4                                                 (Double_ResNet101_mask_rcnn)
# double_swin_DivideSKNet_decrease, _2, _3                                            (mask_rcnn_swin_small_DivideSKNet)        
# double_resnet_50_dcnv2, _2, _3, _4, _5                                              (Double_ResNet50_mask_rcnn)

# double_swin_rebuild_6h, _2, _3, _4, _5, _6, _7                                  6头在swin内部重建的双主干尝试 (mask_rcnn_swin_small)

##  double_swin_DivideSKNet_6h_decrease, _2, _3, _4, _5, _6, _7                   6头-新版decrease+SFNet(mask_rcnn_swin_small_DivideSKNet)

# double_swin_SrcSKNet_6h, _2, _3, _4, _5, _6, _7                                 6头原版SKNet(mask_rcnn_swin_small_SrcSKNet)

# correlation_swin_6h, _2, _3, _4, _5                                             6头四主干(mask_rcnn_swin_small_Correlation)
# correlation_swin_SrcSKNet_6h, _2, _3, _4, _5                                    6头四主干+原版SKNet算法(mask_rcnn_swin_small_Correlation_SrcSKNet)
# correlation_swin_DivideSKNet_6h, _2, _3, _4, _5                                 6头四主干SFNet(mask_rcnn_swin_small_Correlation_DivideSKNet)
# correlation_swin_DivideSKNet_6h_decrease, _2, _3, _4, _5                        6头四主干decrease+SFNet(mask_rcnn_swin_small_Correlation_DivideSKNet)


'''测试特征波段选择效果'''
# double_resnet_50_dcnv2_80_127_206, _2, _3, _4, _5, newLr, _6, _7                    (Double_ResNet50_mask_rcnn_80_127_206)
# double_resnet_50_dcnv2_amplitude_56_80_207, _2, _3, _4, _5                          (Double_ResNet50_mask_rcnn_amplitude_56_80_207)
# double_resnet_50_dcnv2_PCA_56_117_207 , _2, _3, _4, _5                              (Double_ResNet50_mask_rcnn_PCA_56_117_207)
#                                                               
# # #  测试三主干结果 (cfg在输出文件夹后的括号里)
model_cfg_name = "mask_rcnn_swin_small_Correlation_SrcSKNet" + ".py"


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path',
        default=work_path + model_cfg_name)

    parser.add_argument('--work-dir', help='the dir to save logs and models',
        default=R"./work_dirs/" + out_fold)
    parser.add_argument('--resume', nargs='?', type=str, const='auto',
            help='If specify checkpoint path, resume from it, while if not '
            'specify, try to auto resume from the latest checkpoint in the work directory.')
    parser.add_argument('--amp', action='store_true',
            help='enable automatic-mixed-precision training')
    parser.add_argument('--no-validate', action='store_true',
            help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--auto-scale-lr', action='store_true',
        default=False,
            help='whether to auto scale the learning rate according to the '
            'actual batch size and the original batch size.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
            help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. If the value to '
            'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            'Note that the quotation marks are necessary and that no white space '
            'is allowed.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
            help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
