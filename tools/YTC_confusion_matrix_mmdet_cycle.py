# 2023.11 为mmdetr更新
# 杨天成写的自动绘制目标检测混淆矩阵代码
import pandas as pd
import numpy as np
import cv2
import json
# import datetime
# import time
import os
from tqdm import tqdm
import sys

'''
# 统计混淆矩阵前, 先用自己的环境, 把预测结果写到一个json文件里
# 保存格式为:
    [
    {
        "image_name": "20220831_17305000.png",
        "box": [
        290.4693908691406,
        47.391021728515625,
        353.860107421875,
        110.9666976928711
        ],
        "classes": 4,
        "scores": 0.999745786190033
    },
    {
        "image_name": "20220831_17305000.png",
        "box": [
        160.93980407714844,
        445.7268981933594,
        216.11802673339844,
        508.741455078125
        ],
        "classes": 2,
        "scores": 0.9991975426673889
    }
    ]
示例代码:
    results=[]
        # 循环预测每个图片:
        for img in imgs:
            # 循环遍历图片中的每个预测结果:
                result = {}
                result.update({'image_name': image_name})  # 图片名字
                result.update({'classes': classes})        # 类别, 从1开始编号
                result.update({'scores': scores})          # 置信度 
                result.update({'box':box})                 # 预测框, [左上x, 左上y, 右下x, 右下y]
                # 统计每个预测框的信息
                results.append(result)
    print("results", results)
    with open("result.json", 'w') as fw:
        json.dump(results, fw, indent=2)
'''
'''老版本代码, class为1~9, 新版本mmdet为0~8, 因此修改了部分索引'''
'''需要安装的库: pip install xlsxwriter; pip install openpyxl'''

model_name = "correlation_swin_SrcSKNet_6h"  # 组合的模型路径
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

# 真实标签json文件夹的路径
                                                # 2022_9_10_PE_PP_PET_FactoryCut_src, 2022_9_10_PE_PP_PET
label_json_path = R"F:\Keras-Tf-Mask_rcnn_Jiantao\2022_9_10_PE_PP_PET/" + \
                   "Band_73_80_207/Test_copy_paste_9Types"  # Coco_Test_copypaste_full, Band_73_80_207/Test_copy_paste_9Types
                    # Band_73_80_207, New_80_127_206, Band_56_117_207, Band_56_80_207
root_path = R"H:\projects\mmdetr_rebuild"

# 统计最好效果的模型
max_prec, max_recall, max_f1_score, prec_for_f1, recall_for_f1, best_acc, best_recall, best_f1_score = 0, 0, 0, 0, 0, "", "", ""
for i in range(15, 27):  # range(14, 25)  [18, 19, 20, 21, 22, 23, 24, 25, 26]

    prejson_path = root_path + "/work_dirs/" + model_name + "/epoch_"+ str(i) + ".json"
                                                            # epoch_9, epoch_12, latest

    # # # mmdet全波段early_fusion通用标签
    # class_name = {0:"Trans_PET", 1:'Color_PE', 2:'Trans_PP', 3:'White_PET', 4:'White_PE', 5:'Color_PET', 6:'Trans_PE', 7:'White_PP', 8:'Color_PP', 9:'BG'}
    # class_id   = {'Trans_PET':0, 'Color_PE':1, 'Trans_PP':2, 'White_PET':3, 'White_PE':4, 'Color_PET':5, 'Trans_PE':6, 'White_PP':7, 'Color_PP':8, 'BG':9}

    # # mmdet 3波段 Band_73_80_207 double_3band通用标签
    class_name = {0:"Color_PP", 1:'Trans_PET', 2:'Color_PET', 3:'Trans_PP', 4:'White_PE', 5:'White_PET', 6:'Trans_PE', 7:'White_PP', 8:'Color_PE', 9:'BG'}
    class_id   = {'Color_PP':0, 'Trans_PET':1, 'Color_PET':2, 'Trans_PP':3, 'White_PE':4, 'White_PET':5, 'Trans_PE':6, 'White_PP':7, 'Color_PE':8, 'BG':9}

    # # # mmdet 3波段 比值法 New_80_127_206 标签
    # class_name = {0:"Trans_PET", 1:'Color_PE', 2:'Trans_PP', 3:'White_PET', 4:'White_PE', 5:'Color_PET', 6:'Trans_PE', 7:'White_PP', 8:'Color_PP', 9:'BG'}
    # class_id   = {'Trans_PET':0, 'Color_PE':1, 'Trans_PP':2, 'White_PET':3, 'White_PE':4, 'Color_PET':5, 'Trans_PE':6, 'White_PP':7, 'Color_PP':8, 'BG':9}

    # # # mmdet 3波段 PCA Band_56_117_207 标签
    # class_name = {0:"Color_PE", 1:'White_PET', 2:'Trans_PP', 3:'Trans_PE', 4:'White_PE', 5:'White_PP', 6:'Color_PET', 7:'Trans_PET', 8:'Color_PP', 9:'BG'}
    # class_id   = {'Color_PE':0, 'White_PET':1, 'Trans_PP':2, 'Trans_PE':3, 'White_PE':4, 'White_PP':5, 'Color_PET':6, 'Trans_PET':7, 'Color_PP':8, 'BG':9}

    # # # mmdet 3波段 amplitude幅值法 Band_56_80_207 标签
    # class_name = {0:"Trans_PET", 1:'Color_PE', 2:'Trans_PP', 3:'White_PET', 4:'White_PE', 5:'Color_PET', 6:'Trans_PE', 7:'White_PP', 8:'Color_PP', 9:'BG'}
    # class_id   = {'Trans_PET':0, 'Color_PE':1, 'Trans_PP':2, 'White_PET':3, 'White_PE':4, 'Color_PET':5, 'Trans_PE':6, 'White_PP':7, 'Color_PP':8, 'BG':9}


    '''# # 目标矩阵的id顺序, 可以不用改
    # # 按材料
    # # "BG"必须为0, 否则混淆矩阵自动计算将出错
    # dst_name = {0:"BG", 1:"Color_PE", 2:"Trans_PE", 3:"White_PE", 4:"Color_PP", 5:"Trans_PP", 6:"White_PP", 7:"Color_PET", 8:"Trans_PET", 9:"White_PET", 10:""}
    # dst_id =   {"BG":0, "Color_PE":13, "Trans_PE":2, "White_PE":3, "Color_PP":4, "Trans_PP":5, "White_PP":6, "Color_PET":7, "Trans_PET":8, "White_PET":9}
    ''' 
    
    # 按颜色
    # "BG"必须为0, 否则混淆矩阵自动计算将出错
    dst_name = {0:"BG", 1:"Color_PE", 2:"Color_PP", 3:"Color_PET", 4:"Trans_PE", 5:"Trans_PP", 6:"Trans_PET", 7:"White_PE", 8:"White_PP", 9:"White_PET", 10:""}
    dst_id =   {"BG":0, "Color_PE":1, "Color_PP":2, "Color_PET":3, "Trans_PE":4, "Trans_PP":5, "Trans_PET":6, "White_PE":7, "White_PP":8, "White_PET":9}

    num_classes = 9  # 类别总数
    # 0行0列是background, 索引是[行, 列], 最后的行、列增加准确率和召回率
    confusion_matrix = np.zeros(shape=[num_classes + 2, num_classes + 2])
    score_thr = 0.6   # 物体种类置信度阈值, 常用0.6
    iou_thr = 0.7     # 矩形框 IOU置信度阈值

    suffix = ".png"  # 预测图像中, 彩色图为.png, 伪彩色图图.bmp, 光谱图.tif 三通道图像预测同样需要
    prejson = open(prejson_path).read()  # json文件名


    # 判断矩形框是否相交 box是[左上x, 左上y, 右下x, 右下y]
    def mat_inter(box1, box2):
        # 判断两个矩形是否相交
        # box=(xA,yA,xB,yB)
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2

        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)  # 第一个矩形框 长
        sbx = abs(x11 - x12)  # 第二个矩形框 长
        say = abs(y01 - y02)  # 第一个矩形框 宽
        sby = abs(y11 - y12)  # 第二个矩形框 宽
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            # print("相交")
            return True
        else:
            return False


    # 计算两个矩形框的IOU
    def solve_coincide(box1, box2):
        # box = (xA, yA, xB, yB)
        # 计算两个矩形框的重合度
        if mat_inter(box1, box2) == True:
            x01, y01, x02, y02 = box1
            x11, y11, x12, y12 = box2
            col = min(x02, x12) - max(x01, x11)
            row = min(y02, y12) - max(y01, y11)
            intersection = col * row
            area1 = (x02 - x01) * (y02 - y01)
            area2 = (x12 - x11) * (y12 - y11)
            coincide = intersection / (area1 + area2 - intersection)
            # print(coincide)
            return coincide
        else:
            return 0


    last_img_name = '1'
    match_id = []
    miss_det = 0  # 漏识别统计
    wrong_det = 0

    pre_results = json.loads(prejson)
    # tqdm
    for pr in tqdm(range(len(pre_results))):
        result = pre_results[pr]
        # 读取预测结果的信息  box是[左上x, 左上y, 右下x, 右下y]
        image_name, pre_cls_id  = result["image_name"], result["classes"]
        scores,     pre_box     = result["scores"],     result["box"]

        if scores < score_thr:
            continue
        label_json_filename = image_name.replace(suffix, '.json')
        label_json_file = os.path.join(label_json_path, label_json_filename)
        # print("name: ** " + str(image_name))
        # print(label_json_file)
        label_json = open(label_json_file, encoding='utf-8').read()
        label_results = json.loads(label_json)
        # print(image_name)
        if image_name == last_img_name:
            pass
        else:  # 一张图片识别完后, 看这张图片有没有漏识别的, 统计漏识别的, 然后清空match_id
            if last_img_name != '1':
                # print(last_img_name, "匹配到的id: ", match_id)
                last_json_filename = last_img_name.replace(suffix, '.json')
                last_json_file = os.path.join(label_json_path, last_json_filename)
                last_label_json = open(last_json_file).read()
                last_label_results = json.loads(last_label_json)
                for llr in range(len(last_label_results["shapes"])):
                    # print("llr:",llr)
                    if llr in match_id:
                        pass
                    else:
                        cls_name = last_label_results["shapes"][llr]["label"]
                        cls_id = class_id[label_cls_name]
                        confusion_matrix[class_id["BG"], cls_id] += 1  # 漏识别, 原版序号0, 改为BG
                        miss_det += 1  # 漏识别统计
                        # print("漏识别")
                match_id.clear()

        last_img_name = image_name
        get_match = 0
        for lr in range(len(label_results["shapes"])):
            label_result = label_results["shapes"][lr]
            label_cls_name = label_result["label"]
            pts = label_result["points"]
            pts = np.array(pts).astype(int)  # 必须是int, 否则boundingRect会报错
            x, y, w, h = cv2.boundingRect(pts)  # [左上x, 左上y, w, h]
            label_box = [x, y, x+w, y+h]  # 把json文件里的标注轮廓点集转换成box
            pre_cls_name = class_name[pre_cls_id]
            label_cls_id = class_id[label_cls_name]
            # print(solve_coincide(pre_box,label_box))
            if solve_coincide(pre_box, label_box) > iou_thr:
                get_match = 1
                match_id.append(lr)
                if pre_cls_name == label_cls_name:
                    confusion_matrix[pre_cls_id, label_cls_id] += 1  # 预测对
                else:
                    confusion_matrix[pre_cls_id, label_cls_id] += 1  # 预测错
        if get_match == 0:
            confusion_matrix[pre_cls_id, class_id["BG"]] += 1  # 背景被识别成物体, 原版序号0
            wrong_det += 1  # 错识别统计
            # print("错识别背景为物体")

    # 统计最后一张图片是否存在漏识别
    # print(last_img_name, "匹配到的id: ", match_id)
    last_json_filename = last_img_name.replace(suffix, '.json')
    last_json_file = os.path.join(label_json_path, last_json_filename)
    last_label_json = open(last_json_file).read()
    last_label_results = json.loads(last_label_json)
    for llr in range(len(last_label_results["shapes"])):
        # print("llr:", llr)
        if llr in match_id:
            pass
        else:
            cls_name = last_label_results["shapes"][llr]["label"]
            cls_id = class_id[label_cls_name]
            confusion_matrix[class_id["BG"], cls_id] += 1

            miss_det += 1  # 漏识别统计
            # print("最后一张图漏识别")

    # 不转置前: 列是预测的标签, 行是真实的标签, 不方便理解, 故转置
    unsort_numpy = confusion_matrix.T  # 转置后, 每列结果为准确率, 每行结果为召回率

    # 重新按dst_id顺序排列矩阵的行
    mid_numpy = np.zeros((unsort_numpy.shape))
    for id in dst_id:
        # print(str(id) + "  " + str(dst_id[id]))
        mid_numpy[dst_id[id], :] = unsort_numpy[class_id[id], :]

    # 重新按dst_id顺序排列矩阵的列
    dst_numpy = np.zeros((unsort_numpy.shape))
    for id in dst_id:
        # print(str(id) + "  " + str(dst_id[id]))
        dst_numpy[:, dst_id[id]] = mid_numpy[:, class_id[id]]

    # # 计算每类置信度并写入
    for col in range(1, num_classes + 1):
        dst_numpy[num_classes + 1, col] = dst_numpy[col, col] / dst_numpy[:, col].sum()

    # # 计算每类召回率并写入
    for row in range(1, num_classes + 1):
        # print(result_matrix[row, :])
        dst_numpy[row, num_classes + 1] =  dst_numpy[row, row] / dst_numpy[row, :].sum()

    # np增加行列, 以存放平均准确率和平均召回率
    dst_numpy = np.column_stack((dst_numpy, np.zeros((num_classes + 2))))
    dst_numpy = np.row_stack((dst_numpy, np.zeros((num_classes + 3))))
    # 行平均准确率、列平均召回率、F1-score统计
    sum_precision = dst_numpy[num_classes + 1, :].sum() / num_classes
    sum_recall = dst_numpy[:, num_classes + 1].sum() / num_classes
    f1_score = 2 * (sum_precision * sum_recall) / (sum_precision + sum_recall)
    # 统一写入结果
    dst_numpy[num_classes + 2, num_classes + 1] = sum_precision
    dst_numpy[num_classes + 1, num_classes + 2] = sum_recall

    # 对比, 寻找最大的结果
    if sum_precision >= max_prec:
        max_prec = sum_precision
        best_acc = prejson_path
    if sum_recall >= max_recall:
        max_recall = sum_recall
        best_recall = prejson_path
    if  f1_score >= max_f1_score:
        max_f1_score = f1_score
        best_f1_score = prejson_path
        prec_for_f1 = sum_precision
        recall_for_f1 = sum_recall

    # 写入结果
    writer = pd.ExcelWriter(prejson_path[:-5] + '_' + model_name + '_conf_matrix.xlsx')  # 把结果写入Excel
    data = pd.DataFrame(dst_numpy)

    # 重命名轴名称, 将索引替换
    dst_name.setdefault(num_classes + 2, "")  # 将存放平均准确率和平均召回率的标题置空
    data = pd.DataFrame.rename(data, index=dst_name, columns=dst_name)

    # mid = df['采集时间']  # 取备采集时间的值
    # df.pop('采集时间')  # 删除备采集时间
    # df.insert(0, '采集时间', mid)  # 插入采集时间列

    data.to_excel(writer, sheet_name='result')
    worksheet = writer.sheets['result']
    worksheet.set_column('A:Z', 10)

    writer.close()
    print("Results of epoch_"+ str(i))
    print("漏识别统计: ", miss_det)
    print("错识别统计: ", wrong_det)
    # prejson_path[:-4] + '_conf_matrix.xls'

print("最大precision: ", max_prec, best_acc)
print("最大recall:    ", max_recall, best_recall)
print("最大F1-score:  ", max_f1_score, best_f1_score)
print("最大F1-score的precision和recall为:   ", prec_for_f1, "\t", recall_for_f1)
