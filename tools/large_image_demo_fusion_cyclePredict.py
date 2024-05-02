# Copyright (c) OpenMMLab. All rights reserved.
# 用于预测出可以被混淆矩阵计算的结果
"""Perform MMDET inference on large images (as satellite imagery) as:

```shell
wget -P checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth # noqa: E501, E261.

# python demo/large_image_demo.py \
#     demo/large_image.jpg \
#     configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py \
#     checkpoint/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth
```
"""

import os
import random
from argparse import ArgumentParser
from pathlib import Path
import mmcv
import numpy as np
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar  
from mmdet.apis import inference_detector, init_detector
try:
    from sahi.slicing import slice_image
except ImportError:
    raise ImportError('Please run "pip install -U sahi" '
                      'to install sahi first for large image inference.')
from mmdet.registry import VISUALIZERS
from mmdet.utils.large_image import merge_results_by_nms, shift_predictions
from mmdet.utils.misc import get_file_list
import json
import mmengine.fileio as fileio
import tifffile as tiff
import time

for i in range(15, 27):  # range(14, 25) [18, 19, 20, 21, 22, 23, 24, 25, 26]
    # 模型路径、模型名
    model_dir = R'H:\projects\mmdetr_rebuild\work_dirs/'
    model_fold = 'correlation_swin_SrcSKNet_6h'  # 选择存储对应模型文件、config的文件夹
    # double_swin_rebuild, double_swin_rebuild_newpre, double_swin_rebuild_newpre_2
    # double_swin_SrcSKNet, double_swin_SrcSKNet_2, double_swin_SrcSKNet_4
    # double_swin_DivideSKNet, double_swin_DivideSKNet_2
    # correlation_swin_not_full_pre, correlation_swin, correlation_swin_2
    # correlation_swin_DivideSKNet, correlation_swin_DivideSKNet_2, correlation_swin_DivideSKNet_3
    # ##接上 correlation_swin_DivideSKNet_decrease
    # correlation_swin_SrcSKNet, correlation_swin_SrcSKNet_2
    # correlation_swin_SF_SKNet, correlation_swin_SF_SKNet_2, correlation_swin_SF_SKNet_3
    # ##接上 correlation_swin_DivideSKNet_decrease, correlation_swin_DivideSKNet_decrease_2, correlation_swin_DivideSKNet_decrease_3_newLr
    # ##接上 correlation_swin_DivideSKNet_increase, correlation_swin_DivideSKNet_increase_2
    # ##接上 correlation_swin_DivideSKNet_2222, correlation_swin_DivideSKNet_2222_2
    # ...
    # correlation_swin_SrcSKNet_6h, _2, _3, _4, _5
    # correlation_swin_DivideSKNet_6h_decrease, _2, _3, _4, _5


    model_path = model_dir + model_fold  # 组合的模型路径
    # 获取文件夹下保存的参数文件
    for file in os.listdir(model_path):
        if os.path.join(model_path, file).endswith(".py"): config_name = file

    config_file = model_path + '/' + config_name # 组合的改版网络路径


    checkpoint_file = model_path + '/' + 'epoch_'+ str(i) + '.pth'  # 

                                                    # 2022_9_10_PE_PP_PET_FactoryCut_src, 2022_9_10_PE_PP_PET
    test_image_path = R"F:\Keras-Tf-Mask_rcnn_Jiantao\2022_9_10_PE_PP_PET/" + \
                    "Band_73_80_207/Test_copy_paste_9Types"  # Coco_Test_copypaste_full, Band_73_80_207/Test_copy_paste_9Types
                    # Band_73_80_207, New_80_127_206, Band_56_117_207, Band_56_80_207
    output_path = model_path + "/predict_results_epoch_" + str(i)
    # classes = ("Trans_PET","Color_PE","Trans_PP",   # 0~2
    #            "White_PET","White_PE","Color_PET",  # 3~5
    #            "Trans_PE", "White_PP","Color_PP")   # 6~8

    def parse_args():
        parser = ArgumentParser(description='Perform MMDET inference on large images.')
        parser.add_argument('--img', help='Image path, include image file, dir and URL.',
            default=test_image_path)
        parser.add_argument('--config', help='Config file', default=config_file)
        parser.add_argument('--checkpoint', help='Checkpoint file', default=checkpoint_file)
        parser.add_argument('--out-dir', help='Path to output file', default=output_path)
        parser.add_argument('--device', default='cuda:0', help='Device used for inference')
        parser.add_argument('--show', action='store_true', help='Show the detection results')
        parser.add_argument('--tta', action='store_true', help='Whether to use test time augmentation')
        parser.add_argument('--score-thr', type=float, default=0.6, help='Bbox score threshold')  # 0.3
        parser.add_argument('--patch-size', type=int, default=640, help='The size of patches')
        parser.add_argument('--patch-overlap-ratio', type=float, help='Ratio of overlap between two patches',
            default=0.25)
        parser.add_argument('--merge-iou-thr', type=float, help='IoU threshould for merging results',
            default=0.25)
        parser.add_argument('--merge-nms-type', type=str, help='NMS type for merging results',
            default='nms')
        parser.add_argument('--batch-size',type=int, help='Batch size, must greater than or equal to 1',
            default=1)
        parser.add_argument('--debug', action='store_true', help='Export debug results before merging')
        parser.add_argument('--save-patch', action='store_true', help='Save the results of each patch. '
                                                                    'The `--debug` must be enabled.')
        args = parser.parse_args()
        return args


    # 独立出此部分以美化代码
    def client_(file_path, file_client_args, backend_args):
        if file_client_args is not None:
            file_client = fileio.FileClient.infer_client(
                file_client_args, file_path)
            img_bytes = file_client.get(file_path)
        else:
            img_bytes = fileio.get(
                file_path, backend_args=backend_args)
        return img_bytes


    def main():
        args = parse_args()

        config = args.config

        if isinstance(config, (str, Path)):
            config = Config.fromfile(config)
        elif not isinstance(config, Config):
            raise TypeError('config must be a file_path or Config object, '
                            f'but got {type(config)}')
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None

        if args.tta:
            assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
                                        " Can't use tta !"
            assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
                                            "in config. Can't use tta !"
            config.model = ConfigDict(**config.tta_model, module=config.model)
            test_data_cfg = config.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']

            test_data_cfg.pipeline = config.tta_pipeline

        # TODO: TTA mode will error if cfg_options is not set.
        #  This is an mmdet issue and needs to be fixed later.
        # build the model from a config file and a checkpoint file
        model = init_detector(
            config, args.checkpoint, device=args.device, cfg_options={})

        if not os.path.exists(args.out_dir) and not args.show:
            os.mkdir(args.out_dir)
        # init visualizer
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta

        # get file list
        files, source_type = get_file_list(args.img)

        # start detector inference
        print(f'Performing inference on {len(files) / 2} images.... '
            'This may take a while.')
        
        # 统计混淆矩阵前, 先用自己的环境, 把预测结果写到一个json文件里  # 混淆矩阵
        results = []  # 混淆矩阵
        times_name  = []  # 时间+图像名统计
        times_  = 0.0   # 时间统计
        # 开始test文件夹内图像循环
        progress_bar = ProgressBar(len(files) / 2)  # 进度条代码
        for i, file_path in enumerate(files):
            # 忽略tif结尾的文件, 防止被用来可视化, 导致读取失败
            if file_path.endswith('.tif') or file_path.endswith('.bmp'):
                continue

            # 默认参数设置, 暂不修改
            # 代码来自: H:\projects\mmdetr\mmdet\datasets\transforms\loading.py
            file_client_args = None
            imdecode_backend = 'cv2'
            color_type = 'color'
            ignore_empty = False
            suffix = config.test_pipeline[0]['suffix']
            if_fusion = config.test_pipeline[0]['if_fusion']
            backend_args = config.backend_args
            try:
                # 不是融合数据时:
                if if_fusion==False:
                    # 只要不为tif文件, 都仅当单模态RGB图像读取
                    if suffix != 'tif':
                        img_bytes = client_(file_path, file_client_args, backend_args)
                        img = mmcv.imfrombytes(img_bytes, flag=color_type, backend=imdecode_backend)
                    # tif数据, 仅读取单模态NIR图像
                    else:  # suffix == 'tif'
                        spec_ = tiff.imread(file_path[:-3] + suffix)
                        img = (spec_ / 15.6863).round().astype(np.uint8)  # (4000/255=15.6863)
                # 融合数据时:
                else:  # if_fusion==True
                    # 融合数据, 读取RGB和NIR图像并concat
                    if suffix != 'tif':
                        img_bytes = client_(file_path, file_client_args, backend_args)
                        img_ = mmcv.imfrombytes(img_bytes, flag=color_type, backend=imdecode_backend)
                        spec_bytes = client_(file_path[:-3] + suffix, file_client_args, backend_args)
                        spec_ = mmcv.imfrombytes(spec_bytes, flag=color_type, backend=imdecode_backend)
                        # concat
                        img = np.concatenate((img_, spec_), axis=2)
                    else:  # suffix == 'tif'
                        img_bytes = client_(file_path, file_client_args, backend_args)
                        img_ = mmcv.imfrombytes(img_bytes, flag=color_type, backend=imdecode_backend)
                        spec_ = tiff.imread(file_path[:-3] + suffix)
                        spec_ = (spec_ / 15.6863).round().astype(np.uint8)  # (4000/255=15.6863)
                        # concat
                        img = np.concatenate((img_, spec_), axis=2)
                    # else: raise ValueError("Suffix error")

            except Exception as e:
                if ignore_empty:
                    return None
                else:
                    raise e

            # 预测和预测时间
            start = time.perf_counter()
            result_list = inference_detector(model, img)  # 预测结果
            time_ = time.perf_counter() - start

            ignore_num = 2  # 忽略前n张图, 避免错误预测时间
            if i >= ignore_num:
                times_ = times_ + time_
            # import sys
            # sys.exit()
        
            boxes_ = result_list.pred_instances.bboxes.tolist()
            scores_ = result_list.pred_instances.scores.tolist()
            classes_ = result_list.pred_instances.labels.tolist()
            # print(boxes_.shape)
            # print(scores_.shape)
            # print(classes_.shape)
            # print("********************************")

            # # 读取用于可视化的图片
            # visualize_img = mmcv.imread(file_path)
            # 结果可视化并保存至模型目录
            filename = file_path.split("\\")[-1]  # 文件名
            out_file = os.path.join(args.out_dir, filename)

            for num in range(0, len(scores_)):
                # print(boxes_[num])
                # print(scores_[num])
                # print(classes_[num])
                # print("________")
                # 循环预测每个图片:
                result_ = {}  # 混淆矩阵
                result_.update({'image_name': filename})  # 图片名字  # 混淆矩阵
                result_.update({'box':boxes_[num]})            # 预测框, [左上x, 左上y, 右下x, 右下y]  # 混淆矩阵
                result_.update({'scores': scores_[num]})       # 置信度  # 混淆矩阵
                result_.update({'classes': classes_[num]})     # 类别, 从1开始编号  # 混淆矩阵

                # # 可视化结果图
                # visualizer.add_datasample(
                #     filename,
                #     visualize_img,
                #     data_sample=result_list,
                #     draw_gt=False,
                #     show=args.show,
                #     wait_time=0,
                #     out_file=out_file,
                #     pred_score_thr=args.score_thr,
                # )

                # 统计每个预测框的信息    # 混淆矩阵
                results.append(result_)    # 混淆矩阵
            # 保存混淆矩阵至json文件
            with open(checkpoint_file[:-4] + ".json", 'w') as fw:  # 混淆矩阵      
                json.dump(results, fw, indent=2)  # 混淆矩阵

            # 统计预测时间信息
            times_name.append((str(time_*1000) + " ms   " + filename))
            # 打印每张图片时间信息
            # print(time_*1000, "ms \t", filename)
            # 保存预测时间至json文件
            with open(checkpoint_file[:-4] + "_time.json", 'w') as fw:
                json.dump("Mean predict time:" + \
                          str(times_ / ((len(files)/2)-ignore_num) * 1000) + " ms", fw, indent=2)
                json.dump("", fw, indent=2)
                json.dump(times_name, fw, indent=2)

            progress_bar.update()  # 更新进度条

        if not args.show or (args.debug and args.save_patch):
            print_log(
                f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


    if __name__ == '__main__':
        main()
