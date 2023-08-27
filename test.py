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

resize_image_list = []

class SNR_SKF(Node):
    def __init__(self):
        super().__init__("SNR_SKF")

        self.bridge = CvBridge()
        self.i = 1
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_ALL,
                                depth=40,
                                durability=QoSDurabilityPolicy.VOLATILE)
        self.resize_sub = self.create_subscription(Image, 'left/image_resize', self.resize_sub_callback, qos_profile=qos_policy)
        self.snr_publish = self.create_publisher(Image, 'left/image_snr', 200)
        self.timer = self.create_timer(0.2, self.snr_pub_callback)
    
    def resize_sub_callback(self, resize_msg):
        
        # save_folder = './results/{}'.format(opt['name'])
        # GT_folder = osp.join(save_folder, 'images/GT')
        # output_folder = osp.join(save_folder, 'images/output')
        # input_folder = osp.join(save_folder, 'images/input')
        # util.mkdirs(save_folder)
        # util.mkdirs(GT_folder)
        # util.mkdirs(output_folder)
        # util.mkdirs(input_folder)

        # print('mkdir finish')

        # util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
        # logger = logging.getLogger('base')


        # for phase, dataset_opt in opt['datasets'].items():
            
        #     # create_dataset, dataloader from data/__init__.py
        #     val_set = create_dataset(dataset_opt)
        #     val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        #     pbar = util.ProgressBar(len(val_loader))
        #     psnr_rlt = {}  # with border and center frames
        #     psnr_rlt_avg = {}
        #     psnr_total_avg = 0.

        #     ssim_rlt = {}  # with border and center frames
        #     ssim_rlt_avg = {}
        #     ssim_total_avg = 0.

            # for val_data in val_loader:
            #     folder = val_data['folder'][0]
            #     idx_d = val_data['idx']
                
            #     if psnr_rlt.get(folder, None) is None:
            #         psnr_rlt[folder] = []

            #     if ssim_rlt.get(folder, None) is None:
            #         ssim_rlt[folder] = []

            #     LQ = val_data['LQs'].cuda()
            #     nf = val_data['nf'].cuda()
            #     GT = val_data['GT'].cuda()
                
            #     if seg_model is not None:
            #         # a = time.time()
            #         seg_map, seg_feature = seg_model(LQ)

            #         # b = time.time()
            #         # print(b-a)
            #     else:
            #         seg_map, seg_feature = None, None
                
        # change the input from ROS msgs
        # data/utils, dataset_LOLv2.py
        # input to model: LQ, ground truth: GT, processed: nf 
        self.LQ = self.bridge.imgmsg_to_cv2(resize_msg, 'bgr8')
        resize_image_list.append(self.LQ.astype(np.float32) / 255.)

        torch.cuda.empty_cache()
        self.get_logger().info(f'Received_resize_image: {self.i}')
        self.get_logger().info(f'Resize_timestamp: {resize_msg.header.stamp}')
        self.i += 1

    
    def snr_pub_callback(self):

        torch.cuda.empty_cache()
        if opt['seg']:
            seg_model = create_hrnet().cuda()
            seg_model.eval()
        else:
            seg_model = None
        model = create_model(opt)

        time1 = time.time()
        fps = 0.0

        for n in range(len(resize_image_list)):
            if n % 2:
                LQ = np.stack(resize_image_list[:n])[:, :, :, [2, 1, 0]]
                LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(LQ, (0, 3, 1, 2)))).float().cuda()
                print('LQ shape:', LQ.shape)
                nf = LQ[0,:,:,:].permute(1, 2, 0) * 255.0 # Tensor.permute, np.transpose
                nf = nf.detach().cpu()
                nf = cv2.blur(np.float32(nf), (5, 5))
                nf = nf * 1.0 / 255.0
                nf = torch.Tensor(nf).float().permute(2, 0, 1).cuda()
                print('nf shape:', nf.shape)
                GT = LQ # [:, :, :, [2, 1, 0]]
                # GT = torch.from_numpy(np.ascontiguousarray(np.transpose(GT, (0, 3, 1, 2)))).float()
            
                if seg_model is not None:
                    seg_map, seg_feature = seg_model(LQ)
                        
                else:
                    seg_map, seg_feature = None, None
            
                model.feed_data_1(LQ, nf, GT, seg_map, seg_feature)
                model.test4_seg()
                visuals = model.get_current_visuals()
                rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                gt_img = util.tensor2img(visuals['GT'])

                time2 = time.time()
                infer_time = time2 - time1
                fps = fps + (1. / infer_time)

                torch.cuda.empty_cache()

                # if seg_model is not None:
                #     LQ2 = visuals['LQ2']
                #     seg_map, seg_feature = seg_model(LQ2)
                #     model.feed_data_1(LQ, nf, GT, seg_map, seg_feature)
                #     model.test()
                #     visuals = model.get_current_visuals()
                        

                #     rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                #     # gt_img = util.tensor2img(visuals['GT'])  # uint8
                # else:
                #     seg_map, seg_feature = None, None    
                    
                stream_imgs = True
                if stream_imgs:
                    # print(type(rlt_img))
                    rlt_img = cv2.putText(rlt_img.copy(), "InferenceTime = %.2fs/frame" % (infer_time), org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=2)
                    rlt_img = cv2.putText(rlt_img.copy(), "FPS = %.2f" % (fps), org=(0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
                    cv2.imshow('SNR', rlt_img)
                    cv2.waitKey(1)
                    # bayer = cv2.putText(bayer.copy(), "InferenceTime = %.2fs/frame" % (infer_time), org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=2)
                    # bayer = cv2.putText(bayer.copy(), "FPS = %.2f" % (fps), org=(0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
                    # cv2.imshow('SNR', bayer)
                    # cv2.waitKey(1)

                save_imgs = False
                if save_imgs:
                    path_snr = 'home/oz/workspaces/isaac_ros-dev/src/Semantic-Aware-Low-Light-Image-Enhancement/real-time_results/24_Aug_2023/snr_image'
                    path_org = 'home/oz/workspaces/isaac_ros-dev/src/Semantic-Aware-Low-Light-Image-Enhancement/real-time_results/24_Aug_2023/org_image'
                    cv2.imwrite(os.path.join(path_snr, f'SNR_{self.i}.jpg'), rlt_img)
                    cv2.imwrite(os.path.join(path_org, f'org_{self.i}.jpg'),gt_img)
                
                torch.cuda.empty_cache()
                # ROS msgs
                snr_msg = self.bridge.cv2_to_imgmsg(np.array(rlt_img), 'bgr8')
                ros_time = rclpy.clock.Clock().now().to_msg()
                snr_msg.header.frame_id = "left_camera"
                snr_msg.header.stamp = ros_time

                self.snr_publish.publish(snr_msg)

                del resize_image_list[:n]
            
            

def main(args=None):
    rclpy.init(args=args)
    snr_publisher = SNR_SKF()
    rclpy.spin(snr_publisher)
    snr_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
