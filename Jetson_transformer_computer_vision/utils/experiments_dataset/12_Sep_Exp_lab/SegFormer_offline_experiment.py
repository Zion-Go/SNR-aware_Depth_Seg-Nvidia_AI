from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
from datasets import load_dataset
from PIL import Image
from torch import nn
import numpy as np
import cv2
import sys
# from PIL import Image
# import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from ament_index_python.packages import get_package_share_directory
import message_filters
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
import time
import os
import os.path as osp
import glob
import json
from huggingface_hub import hf_hub_download
from torch import nn
import evaluate
from rclpy.executors import MultiThreadedExecutor
import threading

np.set_printoptions(threshold=np.inf)
# from mmseg.core.evaluation import eval_metrics, mean_dice, mean_iou

torch.cuda.empty_cache()

hf_dataset_identifier = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
repo_id = f"{hf_dataset_identifier}"
filename = "config.json"
model_config = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename), "r"))
id2label = model_config["id2label"] # {int(k): v for k, v in id2label.items()}
label2id = model_config["label2id"] # {v: k for k, v in id2label.items()}

num_labels = len(id2label)
print("Labels:", num_labels)

# metric = evaluate.load("mean_iou")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
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

def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
        """Intersection over Union
        Args:
            pred_label (np.ndarray): 2D predict map
            label (np.ndarray): label 2D label map
            num_classes (int): number of categories
            ignore_index (int): index ignore in evaluation
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        n = num_classes
        inds = n * label + pred_label

        mat = np.bincount(inds, minlength=n**2).reshape(n, n)

        return mat

test_path = '/workspaces/isaac_ros-dev/Semantic-Aware-Low-Light-Image-Enhancement/experiments_dataset/29_Aug_snr/low_light_dataset/right'
output_path = '/workspaces/isaac_ros-dev/Semantic-Aware-Low-Light-Image-Enhancement/experiments_dataset/tables_figures/day_seg'

count = 0
jpg_list = sorted(glob.glob(osp.join(test_path, '*.jpg')))

for image in jpg_list:
    jpg_name = osp.basename(image)
    image = cv2.imread(image)
    # print(type(image))
    # if count == 1:
    #      break
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
        
        # print(logits[0])
        # print('logits:', type(logits), logits.shape)
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
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array(cityscapes_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
            
            # print(label) # 0-18

        # Convert to BGR
        color_seg = color_seg[..., ::-1]
        # print('seg:', seg.shape, seg) # .tolist()
        # print('color_seg:', color_seg.shape)

        # Show image + mask
        img_seg = np.array(image) * 0.5 + color_seg * 0.5
        img_seg = img_seg.astype(np.uint8)


        # Save the outputs
        cv2.imwrite(f'{output_path}/day_seg_{count}.jpg', img_seg)
        print("img_seg saved to: " + output_path)
        count += 1


    # https://github.com/NVlabs/SegFormer/blob/master/tests/test_metrics.py
    # https://huggingface.co/blog/fine-tune-segformer
        
 
    

