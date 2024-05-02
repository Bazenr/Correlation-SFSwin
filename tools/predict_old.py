from mmdet.apis import init_detector, inference_detector  #, show_result_pyplot
import os
from tqdm import tqdm
import sys
# import numpy as np
import json
from mmdet.apis import DetInferencer
import mmengine

# 原版网络


# 改版网络
config_file = R'H:\projects\mmdetr_rebuild\configs\swin/' + \
                'mask-rcnn_swin-t-p4-w7_fpn_1x_coco' + '.py'

                
# 模型路径、模型名
model_dir = R'H:\projects\mmdetr_rebuild\\work_dirs'
fold = 'early_fusion'  # 模型文件夹
model_path = model_dir + '/' + fold  # 组合的模型路径

checkpoint_file = model_path + '/' + 'epoch_12.pth'
                                    # epoch_3, , epoch_7, epoch_9, latest

test_image_path = R"F:\Keras-Tf-Mask_rcnn_Jiantao\2022_9_10_PE_PP_PET_FactoryCut_src/" + \
                   "/Train_Val_copypaste_gather"  # 
device = 'cuda:0'
# suffix_1为上层3通道图, suffix_2为下层通道图
suffix_1 = ".png"  # 预测图像中, 彩色图为.png, 灰度图.bmp, 光谱图.tif
num_classes = 9


def main():     
    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)
    # 统计混淆矩阵前, 先用自己的环境, 把预测结果写到一个json文件里  # 混淆矩阵
    results = []  # 混淆矩阵
    print("Test img path: " + test_image_path)
    # 推理演示图像
    filelist = get_filelist(test_image_path, [])
    for file in tqdm(filelist):
        if file.endswith(suffix_1):
            (filepath, filename) = os.path.split(file)  # 绝对路径中提取文件名

            inference_result = inference_detector(model, file)

            # 保存图像结果至image_result文件夹
            # saving_path = mkdir(os.path.join(model_path, 'image_result'))
            # model.show_result(file, inference_result, out_file=os.path.join(saving_path, filename))

            # # 可视化图像结果
            # show_result_pyplot(model, file, inference_result)
            print(type(inference_result))
            for class_ in range(num_classes):  # result[0][0~8]分别为每类的box结果, 4个坐标+置信度
                for isinstance in range(len(inference_result[0][class_])):
                    # print("Class:" + str(class_) + "  " + str(inference_result[0][class_][isinstance]))  # 打印单个结果测试
                    
                    boxes_ = inference_result[0][class_][isinstance][0:4].tolist()  # 混淆矩阵
                    scores_ = inference_result[0][class_][isinstance][4].tolist()  # 混淆矩阵
                    classes_ = class_ + 1  # 混淆矩阵

                    # 循环预测每个图片:
                    result_ = {}  # 混淆矩阵
                    result_.update({'image_name': filename})  # 图片名字  # 混淆矩阵
                    result_.update({'box':boxes_})            # 预测框, [左上x, 左上y, 右下x, 右下y]  # 混淆矩阵
                    result_.update({'scores': scores_})       # 置信度  # 混淆矩阵
                    result_.update({'classes': classes_})     # 类别, 从1开始编号  # 混淆矩阵
                    # 统计每个预测框的信息    # 混淆矩阵
                    results.append(result_)    # 混淆矩阵

            # 保存混淆矩阵至json文件
            with open(checkpoint_file[:-4] + ".json", 'w') as fw:  # 混淆矩阵      
                json.dump(results, fw, indent=2)  # 混淆矩阵

            # break可以使程序仅保存第一个结果, 测试用
            # break


#  遍历文件夹及其子文件夹中的文件名, 并存储在一个列表中
def get_filelist(dir_, file_list):
    """
    示例: doc_list = get_filelist(outputdir, [])

    遍历文件夹及其子文件夹中的文件，并存储在一个列表中
    输入文件夹路径、空文件列表[]
    返回 文件列表Filelist,包含文件名（完整路径）
    原文链接: https://blog.csdn.net/weixin_41521681/article/details/92768157
    """
    if os.path.isfile(dir_):
        file_list.append(dir_)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir_):
        for s in os.listdir(dir_):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            #     continue
            new_dir = os.path.join(dir_, s)
            get_filelist(new_dir, file_list)

    return file_list


# 判断所输入路径是否存在文件夹, 不存在则新建文件夹
def mkdir(path):
    """原文链接: https: // blog.csdn.net / vip_lvkang / article / details / 76906718"""
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("Making path: %s", path)
    return path


if __name__ == "__main__":
    main()
