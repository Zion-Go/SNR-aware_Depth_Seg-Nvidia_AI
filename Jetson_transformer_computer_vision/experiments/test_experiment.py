import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
from PIL import Image

import utils.util as util
import data.util as data_util
from models import create_model

import os.path as osp
import logging
import argparse
from collections import OrderedDict
from models.hrseg_model import create_hrnet

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import time
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

torch.cuda.empty_cache()
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# Arguments options
parser = argparse.ArgumentParser()
# parser.add_argument('-opt', default="options/test/LOLv1_seg.yml", type=str, help='Path to options YMAL file.')
parser.add_argument('-opt', default="options/test/LOLv2_real_seg.yml", type=str, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

def main():
    
    # below has: if save_imgs:
    save_imgs = True
    
    if opt['seg']:
        seg_model = create_hrnet().cuda()
        seg_model.eval()
    else:
        seg_model = None
    model = create_model(opt)
    
    save_folder = './results/{}'.format(opt['name'])
    GT_folder = osp.join(save_folder, 'images/GT')
    output_folder = osp.join(save_folder, 'images/output')
    input_folder = osp.join(save_folder, 'images/input')
    util.mkdirs(save_folder)
    util.mkdirs(GT_folder)
    util.mkdirs(output_folder)
    util.mkdirs(input_folder)

    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')


    for phase, dataset_opt in opt['datasets'].items():
        
        # create_dataset, dataloader from data/__init__.py
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}  # with border and center frames
        psnr_rlt_avg = {}
        psnr_total_avg = 0.

        ssim_rlt = {}  # with border and center frames
        ssim_rlt_avg = {}
        ssim_total_avg = 0.

        for val_data in val_loader:
            folder = val_data['folder'][0]
            idx_d = val_data['idx']
            
            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []

            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []

            LQ = val_data['LQs'].cuda()
            nf = val_data['nf'].cuda()
            # GT = val_data['GT'].cuda()
            
            if seg_model is not None:
                # a = time.time()
                seg_map, seg_feature = seg_model(LQ)

                # b = time.time()
                # print(b-a)
            else:
                seg_map, seg_feature = None, None
            
            # change the input from ROS msgs
            # data/utils, dataset_LOLv2.py
            # input to model: LQ, ground truth: GT, processed: nf 
            # LQ = 
            # nf = LQ.permute(1, 2, 0).numpy() * 255.0
            # nf = cv2.blur(nf, (5, 5))
            # nf = nf * 1.0 / 255.0
            # nf = torch.Tensor(nf).float().permute(2, 0, 1)
            # GT = LQ[:, :, :, [2, 1, 0]]
            # GT = torch.from_numpy(np.ascontiguousarray(np.transpose(GT, (0, 3, 1, 2)))).float()
            
            # can change the functions 
            model.feed_data_2(LQ, nf, seg_map, seg_feature)
            model.test4_seg()
            visuals = model.get_current_visuals()
            # break
            # model.feed_data(val_data)

            rlt_img = util.tensor2img(visuals['rlt'])  # uint8
            # gt_img = util.tensor2img(visuals['GT'])  # uint8

            if seg_model is not None:
                # a = time.time()
                LQ2 = visuals['LQ2']
                seg_map, seg_feature = seg_model(LQ2)
                model.feed_data_2(LQ, nf, seg_map, seg_feature)
                model.test4_seg()
                visuals = model.get_current_visuals()
                # model.feed_data(val_data)

                # LQ2 = visuals['LQ2']
                # seg_map, seg_feature = seg_model(LQ2)
                # model.feed_data_1(LQ, nf, GT, seg_map, seg_feature)
                # model.test()
                # visuals = model.get_current_visuals()
                # # model.feed_data(val_data)

                rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                # gt_img = util.tensor2img(visuals['GT'])  # uint8
                # b = time.time()
                # print(b-a)
            else:
                seg_map, seg_feature = None, None


            mid_ix = dataset_opt['N_frames'] // 2
            input_img = util.tensor2img(visuals['LQ'][mid_ix])
            
            stream_imgs = False
            if stream_imgs:
                cv2.imshow('SNR', rlt_img)
                cv2.waitKey(1)

                # ROS msgs
                # seg_msg = self.bridge.cv2_to_imgmsg(np.array(image_list[n]), 'bgr8')
                # ros_time = rclpy.clock.Clock().now().to_msg()
                # seg_msg.header.frame_id = "left_camera"
                # seg_msg.header.stamp = ros_time

                # self.seg_publish.publish(seg_msg)

                # del image_list[:n]

            if save_imgs:
                try:
                    # tag = '{}.{}'.format(val_data['folder'], idx_d[0].replace('/', '-'))
                    tag = '{}'.format(val_data['folder'])[2:-6]
                    print(osp.join(output_folder, '{}.png'.format(tag)))
                    cv2.imwrite(osp.join(output_folder, '{}.png'.format(tag)), rlt_img)
                    # cv2.imwrite(osp.join(GT_folder, '{}.png'.format(tag)), gt_img)

                    cv2.imwrite(osp.join(input_folder, '{}.png'.format(tag)), input_img)

                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()
'''
            # calculate PSNR
            psnr = util.calculate_psnr(rlt_img, gt_img)
            psnr_rlt[folder].append(psnr)

            ssim = util.calculate_ssim(rlt_img, gt_img)
            # ssim = 0
            ssim_rlt[folder].append(ssim)

            pbar.update('Test {} - {}'.format(folder, idx_d))
        
        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]

        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]

        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
        
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
        
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        psnr_all = 0
        psnr_count = 0
        
        for k, v in psnr_rlt.items():
            psnr_all += sum(v)
            psnr_count += len(v)
        psnr_all = psnr_all * 1.0 / psnr_count
        print('psnr_all:', psnr_all)

        ssim_all = 0
        ssim_count = 0
        
        for k, v in ssim_rlt.items():
            ssim_all += sum(v)
            ssim_count += len(v)
        ssim_all = ssim_all * 1.0 / ssim_count
        print('ssim_all:', ssim_all)
'''

if __name__ == '__main__':
    main()
