from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch import nn
import evaluate

from PIL import Image
from torch import nn
import numpy as np
import cv2
import sys
# from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sn

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
import message_filters
from rclpy.executors import MultiThreadedExecutor

import math
import time
import os
import os.path as osp
import glob
import json
import threading
# from mmseg.core.evaluation import eval_metrics, mean_dice, mean_iou

torch.cuda.empty_cache()

hf_dataset_identifier = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
repo_id = f"{hf_dataset_identifier}"
filename = "config.json"
model_config = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename), "r"))
id2label = model_config["id2label"] # {int(k): v for k, v in id2label.items()}
label2id = model_config["label2id"] # {v: k for k, v in id2label.items()}

num_labels = len(id2label)
print("---------->Labels:", num_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
model.to(device)

# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""

    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

def cityscapes_palette():
    '''
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    '''
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    return [[128, 64,128],[244, 35,232],[70, 70, 70],[102,102,156],
            [190,153,153],[153,153,153],[250,170, 30],
            [220,220,  0],[107,142, 35],[152,251,152],[70,130,180],[220, 20, 60],[255,  0,  0],
            [0,  0,142],[0,  0, 70],[0, 60,100],[0, 80,100],[0,  0,230],
            [119, 11, 32]]

def get_confusion_matrix(pred_label, gt_label, num_classes):
    """Intersection over Union
    Args:
        pred_label (np.ndarray): 2D predict map
        label (np.ndarray): label 2D label map
        num_classes (int): number of categories
        ignore_index (int): index ignore in evaluation
    """

    # mask = (label != ignore_index)
    # pred_label = pred_label[mask]
    # label = label[mask]
    # pred_label = np.unique(pred_label)
    # gt_label = np.unique(gt_label)
    pred_label = pred_label.flatten()
    gt_label = gt_label.flatten()
    
    # if len(pred_label) != len(gt_label): # len(arr[0])
    #     print('not equally matching labels')
    #     min_size = min(pred_label.size, gt_label.size)
    #     inds = pred_label[:min_size] + gt_label[:min_size]
    #     print(f'truncate the labels to {min_size}')
    # else:
    inds = num_classes * gt_label + pred_label

    mat = np.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes) # bincount works with a 1d array; it raises this error if given a scalar.

    return mat

def _fast_hist(label_true, label_pred, n_class): # confusion matrix
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += get_confusion_matrix(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def record_score(score, save_path, iou_type):
    score_list = []

    for i in range(21):
        score_list.append(score['Class IoU'][i])
        aveJ = score['Mean IoU']

    with open('{}/iou_{}.txt'.format(save_path, iou_type), 'w') as f:
        for num, cls_iou in enumerate(score_list):
            print('class {:2d} {:12} IU {:.2f}'.format(num, CAT_LIST[num], round(cls_iou, 3)))
            f.write('class {:2d} {:12} IU {:.2f}'.format(num, CAT_LIST[num], round(cls_iou, 3)) + '\n')
        print('meanIOU: ' + str(aveJ) + '\n')
        f.write('meanIOU: ' + str(aveJ) + '\n')

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def update(val, n=1):
    val = val
    sum += val * n
    count += n
    avg = sum / count

def cal_acc(pred, target, classes):
    intersection_meter = 0
    union_meter = 0
    target_meter = 0

    intersection, union, target = intersectionAndUnion(pred, target, classes)
    intersection_meter.update(intersection)
    union_meter.update(union)
    target_meter.update(target)
    accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
    logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return {
        "Pixel Accuracy": allAcc,
        "Mean Accuracy": mAcc,
        "Mean IoU": mIoU,
    }

test_path = '/workspaces/isaac_ros-dev/src/Semantic-Aware-Low-Light-Image-Enhancement/experiments_dataset/29_Aug_snr/experiments_pair_compare_seg'
output_path = '/workspaces/isaac_ros-dev/src/Semantic-Aware-Low-Light-Image-Enhancement/experiments_dataset/28_Aug_seg+snr/experiment_pair_seg'

count = 0
jpg_list = sorted(glob.glob(osp.join(test_path, '*.jpg')))

for image in jpg_list:
    jpg_name = osp.basename(image)
    image = cv2.imread(image)
    # print(type(image))

    with torch.no_grad():
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)

        outputs = model(pixel_values)
        logits = outputs.logits
        # loss = outputs.loss # for training stage

        # SemanticSegmenterOutput:
        # loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification (or regression if config.num_labels==1) loss.
        # logits (torch.FloatTensor of shape (batch_size, config.num_labels, logits_height, logits_width)) — Classification scores for each pixel.
        # rescale logits to original image size
        logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                        size=image.shape[0:2], # (height, width)
                        mode='bilinear',
                        align_corners=False)
        # print('------>logits:', type(logits), logits.shape)
        # metrics for training stage
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        # metrics = metric.compute(
        #         predictions=pred_labels,
        #         references=labels,
        #         num_labels=len(id2label),
        #         ignore_index=0,
        #         reduce_labels=feature_extractor.do_reduce_labels,
        #     )

        torch.cuda.empty_cache()

        # apply argmax on the class dimension
        seg = logits.argmax(dim=1)[0]
        # print(type(logits), seg.shape) # tensor, 400x600
        seg_array = np.array(seg)
        print(f'----------->seg_array labels: {np.unique(seg_array)}')

        # get gt 2D array map
        with open('night_8.json') as f:
            data = json.load(f)

        obj = {k:v for k, v in data.items() if k.startswith('objects')}
        slice = list(obj.values())[0]

        gt_label_map = np.zeros((600, 400))

        for i in range(len(slice)):
            group = {k:v for k, v in slice[i].items() if k.startswith('group')}
            gt_seg = {k:v for k, v in slice[i].items() if k.startswith('segmentation')}
            seg_coor = list(gt_seg.values())
            label_index = list(group.values())[0]

            for j in range(len(seg_coor[0])):
                # print(f'x: {int(seg_coor[0][j][0])}, y: {int(seg_coor[0][j][1])}, {label_index}')
                gt_label_map[int(seg_coor[0][j][0])][int(seg_coor[0][j][1])] = label_index
                # print(f'label_classes: {np.unique(gt_label_map)}')

        # print('------------>confusion mat:', type(seg_array), seg_array.shape, type(gt_label_map), np.array(gt_label_map).transpose().shape)
        # compute confusion matrix
        confusion_matrix = get_confusion_matrix(np.array(seg_array, dtype=np.int), np.array(gt_label_map, dtype=np.int).transpose(), num_labels)
        # plt.figure(figsize = (10,7))
        # sn.heatmap(confusion_matrix, annot=True)
        # plt.show()

        # compute IoU
        # score = scores(np.array(seg_array, dtype=np.int), np.array(gt_label_map, dtype=np.int).transpose(), num_labels)
        # print(score)

        mean_iou = evaluate.load("mean_iou")
        results = mean_iou.compute(predictions=np.array(seg_array, dtype=np.int), references=np.array(gt_label_map, dtype=np.int).transpose(), num_labels=19)
        print('mmseg:', results)

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array(cityscapes_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
            
            # print(label) # 0-18

        # Convert to BGR
        color_seg = color_seg[..., ::-1]
        # print('color_seg:', color_seg.shape)

        # Show image + mask
        img_seg = np.array(image) * 0.5 + color_seg * 0.5
        img_seg = img_seg.astype(np.uint8)


        # Save the outputs
        # cv2.imwrite(f'{output_path}/{jpg_name}', img_seg)
        # print("img_seg saved to: " + output_path)
        count += 1
    if count == 2:
        break


    # https://github.com/NVlabs/SegFormer/blob/master/tests/test_metrics.py
    # https://huggingface.co/blog/fine-tune-segformer
        
    # from torchvision.transforms import ColorJitter
    # from transformers import SegformerFeatureExtractor

    # feature_extractor = SegformerFeatureExtractor()
    # jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 
    
    
    # def test_metrics():
    #     pred_size = (10, 30, 30)
    #     num_classes = 19
    #     ignore_index = 255
    #     results = np.random.randint(0, num_classes, size=pred_size)
    #     label = np.random.randint(0, num_classes, size=pred_size)
    #     label[:, 2, 5:10] = ignore_index
    #     all_acc, acc, iou = eval_metrics(
    #         results, label, num_classes, ignore_index, metrics='mIoU')
    #     all_acc_l, acc_l, iou_l = legacy_mean_iou(results, label, num_classes,
    #                                             ignore_index)
    #     assert all_acc == all_acc_l
    #     assert np.allclose(acc, acc_l)
    #     assert np.allclose(iou, iou_l)

    #     all_acc, acc, dice = eval_metrics(
    #         results, label, num_classes, ignore_index, metrics='mDice')
    #     all_acc_l, acc_l, dice_l = legacy_mean_dice(results, label, num_classes,
    #                                                 ignore_index)
    #     assert all_acc == all_acc_l
    #     assert np.allclose(acc, acc_l)
    #     assert np.allclose(dice, dice_l)

    #     all_acc, acc, iou, dice = eval_metrics(
    #         results, label, num_classes, ignore_index, metrics=['mIoU', 'mDice'])
    #     assert all_acc == all_acc_l
    #     assert np.allclose(acc, acc_l)
    #     assert np.allclose(iou, iou_l)
    #     assert np.allclose(dice, dice_l)

    #     results = np.random.randint(0, 5, size=pred_size)
    #     label = np.random.randint(0, 4, size=pred_size)
    #     all_acc, acc, iou = eval_metrics(
    #         results,
    #         label,
    #         num_classes,
    #         ignore_index=255,
    #         metrics='mIoU',
    #         nan_to_num=-1)
    #     assert acc[-1] == -1
    #     assert iou[-1] == -1

    #     all_acc, acc, dice = eval_metrics(
    #         results,
    #         label,
    #         num_classes,
    #         ignore_index=255,
    #         metrics='mDice',
    #         nan_to_num=-1)
    #     assert acc[-1] == -1
    #     assert dice[-1] == -1

    #     all_acc, acc, dice, iou = eval_metrics(
    #         results,
    #         label,
    #         num_classes,
    #         ignore_index=255,
    #         metrics=['mDice', 'mIoU'],
    #         nan_to_num=-1)
    #     assert acc[-1] == -1
    #     assert dice[-1] == -1
    #     assert iou[-1] == -1

    # 

