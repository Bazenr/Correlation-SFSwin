# Copyright (c) OpenMMLab. All rights reserved.
# 用于预测出可以被混淆矩阵计算的结果
"""Perform MMDET inference on large images (as satellite imagery) as:

```shell
wget -P checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth # noqa: E501, E261.

python demo/large_image_demo.py \
    demo/large_image.jpg \
    configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py \
    checkpoint/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth
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
import time


# 模型路径、模型名
model_dir = R'H:\projects\mmdetr_rebuild\\work_dirs/'
fold = 'RGB'  # 选择存储对应模型文件、config的文件夹
        # RGB, early_fusion
config_name = 'mask-rcnn_swin-t-p4-w7_fpn_1x_coco' + '.py'
model_path = model_dir + fold  # 组合的模型路径
config_file = model_path + '/' + config_name # 组合的改版网络路径

checkpoint_file = model_path + '/' + 'epoch_12.pth'  # 

test_image_path = R"F:\Keras-Tf-Mask_rcnn_Jiantao\2022_9_10_PE_PP_PET_FactoryCut_src/" + \
                   "/Coco_Test_copypaste_full"  # TestFor_mmdetr, Coco_Test_copypaste_full
output_path = model_path + "/predict_results"
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
    parser.add_argument('--score-thr', type=float, default=0.3, help='Bbox score threshold')
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


def main():
    args = parse_args()

    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
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
    print(f'Performing inference on {len(files)} images.... '
          'This may take a while.')
    
    # 统计混淆矩阵前, 先用自己的环境, 把预测结果写到一个json文件里  # 混淆矩阵
    results = []  # 混淆矩阵
    # 开始test文件夹内图像循环
    progress_bar = ProgressBar(len(files))
    for file in files:
        # 忽略tif结尾的文件, 防止被用来可视化, 导致读取失败
        if file.endswith('.tif'):
            continue
        # read image
        img = mmcv.imread(file)

        # arrange slices
        height, width = img.shape[:2]
        # print(type(img))
        # print(img.shape)
        # print("********************************")
        sliced_image_object = slice_image(
            img,
            slice_height=args.patch_size,
            slice_width=args.patch_size,
            auto_slice_resolution=False,
            overlap_height_ratio=args.patch_overlap_ratio,
            overlap_width_ratio=args.patch_overlap_ratio,
        )
        # print(type(sliced_image_object))
        # print(sliced_image_object)
        # print("********************************")
        # import sys
        # sys.exit()
        # perform sliced inference
        slice_results = []
        start = 0
        while True:
            # prepare batch slices
            end = min(start + args.batch_size, len(sliced_image_object))
            images = []
            for sliced_image in sliced_image_object.images[start:end]:
                images.append(sliced_image)

            # 需要将后面其他部分注释以获取真实预测时间
            # start = time.perf_counter()
            slice_results.extend(inference_detector(model, images))
            # time_ = time.perf_counter() - start
            # print(time_*1000, "ms \t")
            
            if end >= len(sliced_image_object):
                break
            start += args.batch_size

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)

        img = mmcv.imconvert(img, 'bgr', 'rgb')
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        # export debug images
        if args.debug:
            # export sliced image results
            name, suffix = os.path.splitext(filename)

            shifted_instances = shift_predictions(
                slice_results,
                sliced_image_object.starting_pixels,
                src_image_shape=(height, width))
            merged_result = slice_results[0].clone()
            merged_result.pred_instances = shifted_instances

            debug_file_name = name + '_debug' + suffix
            debug_out_file = None if args.show else os.path.join(
                args.out_dir, debug_file_name)
            visualizer.set_image(img.copy())

            debug_grids = []
            for starting_point in sliced_image_object.starting_pixels:
                start_point_x = starting_point[0]
                start_point_y = starting_point[1]
                end_point_x = start_point_x + args.patch_size
                end_point_y = start_point_y + args.patch_size
                debug_grids.append(
                    [start_point_x, start_point_y, end_point_x, end_point_y])
            debug_grids = np.array(debug_grids)
            debug_grids[:, 0::2] = np.clip(debug_grids[:, 0::2], 1,
                                           img.shape[1] - 1)
            debug_grids[:, 1::2] = np.clip(debug_grids[:, 1::2], 1,
                                           img.shape[0] - 1)

            palette = np.random.randint(0, 256, size=(len(debug_grids), 3))
            palette = [tuple(c) for c in palette]
            line_styles = random.choices(['-', '-.', ':'], k=len(debug_grids))
            visualizer.draw_bboxes(
                debug_grids,
                edge_colors=palette,
                alpha=1,
                line_styles=line_styles)
            visualizer.draw_bboxes(
                debug_grids, face_colors=palette, alpha=0.15)

            visualizer.draw_texts(
                list(range(len(debug_grids))),
                debug_grids[:, :2] + 5,
                colors='w')

            visualizer.add_datasample(
                debug_file_name,
                visualizer.get_image(),
                data_sample=merged_result,
                draw_gt=False,
                show=args.show,
                wait_time=0,
                out_file=debug_out_file,
                pred_score_thr=args.score_thr,
            )

            if args.save_patch:
                debug_patch_out_dir = os.path.join(args.out_dir,
                                                   f'{name}_patch')
                for i, slice_result in enumerate(slice_results):
                    patch_out_file = os.path.join(
                        debug_patch_out_dir,
                        f'{filename}_slice_{i}_result.jpg')
                    image = mmcv.imconvert(sliced_image_object.images[i],
                                           'bgr', 'rgb')

                    visualizer.add_datasample(
                        'patch_result',
                        image,
                        data_sample=slice_result,
                        draw_gt=False,
                        show=False,
                        wait_time=0,
                        out_file=patch_out_file,
                        pred_score_thr=args.score_thr,
                    )

        image_result = merge_results_by_nms(
            slice_results,
            sliced_image_object.starting_pixels,
            src_image_shape=(height, width),
            nms_cfg={
                'type': args.merge_nms_type,
                'iou_threshold': args.merge_iou_thr
            })

        visualizer.add_datasample(
            filename,
            img,
            data_sample=image_result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr,
        )
        # progress_bar.update()

        # print(filename)
        boxes_ = image_result.pred_instances.bboxes.tolist()
        scores_ = image_result.pred_instances.scores.tolist()
        classes_ = image_result.pred_instances.labels.tolist()
        # print(boxes_.shape)
        # print(scores_.shape)
        # print(classes_.shape)
        # print("********************************")
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

            # import sys
            # sys.exit()

            # 统计每个预测框的信息    # 混淆矩阵
            results.append(result_)    # 混淆矩阵
        # 保存混淆矩阵至json文件
        with open(checkpoint_file[:-4] + ".json", 'w') as fw:  # 混淆矩阵      
            json.dump(results, fw, indent=2)  # 混淆矩阵

    if not args.show or (args.debug and args.save_patch):
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
