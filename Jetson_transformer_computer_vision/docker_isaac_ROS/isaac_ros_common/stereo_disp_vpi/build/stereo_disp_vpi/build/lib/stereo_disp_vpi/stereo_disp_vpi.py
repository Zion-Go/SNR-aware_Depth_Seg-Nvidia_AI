import cv2
import sys
# import vpi
import numpy as np
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
from stereo_disp_vpi import calibration_configs
sys.path.append("/workspaces/isaac_ros-dev/stereo_vpi_pkg/src/stereo_disp_vpi/stereo_disp_vpi/calibration_configs.py")
import time

def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print(f'==============================\nPixel coordinates: x = {x}, y = {y}')
        # print("World coordinates:", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("World coordinates:", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0]**2 + threeD[y][x][1]**2 + threeD[y][x][2]**2)
        distance = distance / 1000.0  # mm -> m
        print("Distance:", distance, "m\n==============================")

# wls filter
def wls_filter(stereo, imgL, imgR):
    
    left_matcher = stereo
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher) 
    # FILTER Parameters
    lmbda = 40000
    visual_multiplier = 4
    sigma = 0.8
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put imgL here
    # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg) # map to 0-255
    return filteredImg

class stereo_disp_vpi(Node):
    def __init__(self):
        super().__init__("stereo_disp_vpi")
        
        self.bridge = CvBridge()
        self.i = 1
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_ALL,
                                depth=20,
                                durability=QoSDurabilityPolicy.VOLATILE)
        
        self.imgl_raw_suber = message_filters.Subscriber(self, Image, "/left/image_raw", qos_profile=qos_policy)
        self.imgr_raw_suber = message_filters.Subscriber(self, Image, "/right/image_raw", qos_profile=qos_policy)
        
        sync_raw = message_filters.ApproximateTimeSynchronizer(
        [self.imgl_raw_suber, self.imgr_raw_suber],
        queue_size=200,
        slop=0.1,
        allow_headerless=False)
        sync_raw.registerCallback(self.sync_raw_callback)

        self.disp_publish = self.create_publisher(Image, '/disparity_stereo', 20)
        self.timer = self.create_timer(0.2, self.disp_pub_callback)
        self.disp_sub = self.create_subscription(Image, '/disparity_stereo', self.disp_sub_callback, qos_profile=qos_policy)
           
        self.left_raw_list, self.right_raw_list = [], []


    def sync_raw_callback(self, imgl_raw_suber, imgr_raw_suber):
        ros_time = rclpy.clock.Clock().now().to_msg()
        imgl_raw_suber.header.stamp = ros_time
        imgr_raw_suber.header.stamp = ros_time
        # print(imgl_rect_suber.header.stamp, imgr_rect_suber.header.stamp)

        # do mono preprocessing here
        if imgl_raw_suber.header.frame_id == "left_camera":
            self.left_raw = self.bridge.imgmsg_to_cv2(imgl_raw_suber, "mono8")
            self.left_raw_list.append(self.left_raw)
            # print('left raw', len(self.left_raw_list))
            # print('left raw', type(self.left_raw))
        if imgr_raw_suber.header.frame_id == "right_camera":
            self.right_raw = self.bridge.imgmsg_to_cv2(imgr_raw_suber, "mono8")
            self.right_raw_list.append(self.right_raw)
            # print('right raw', len(self.right_raw_list))
            # print('right raw', type(self.right_raw))
        

    def image_rect_l_callback(self, rect_msg):
        if rect_msg.header.frame_id == "left_camera":
            self.left_rect = self.bridge.imgmsg_to_cv2(rect_msg, "mono8")
            # print('leftrect', type(self.left_rect))
    
    def image_rect_r_callback(self, rect_msg):
        if rect_msg.header.frame_id == "right_camera":
            self.right_rect = self.bridge.imgmsg_to_cv2(rect_msg, "mono8")
            # print('rightrect', type(self.right_rect))

    def disp_pub_callback(self):
        
        time1 = time.time()
        
        ### disp SGBM
        num = 1
        blockSize = 11
        img_channels = 3
        self.stereo = cv2.StereoSGBM_create(minDisparity=-1,
                                numDisparities=16*num,
                                blockSize=blockSize,
                                P1=8 * img_channels * blockSize * blockSize,
                                P2=32 * img_channels * blockSize * blockSize,
                                disp12MaxDiff=2,
                                preFilterCap=1,
                                uniquenessRatio=6,
                                speckleWindowSize=80,
                                speckleRange=1,
                                mode=cv2.STEREO_SGBM_MODE_SGBM)
        

        ave_depth1, ave_depth2, ave_depth3 = [], [], []

        ### opencv remap stereorectify to get img rect
        for self.left_image_raw, self.right_image_raw in zip(self.left_raw_list, self.right_raw_list):
            

            self.left_rect = cv2.remap(self.left_image_raw, calibration_configs.left_map1, calibration_configs.left_map2, cv2.INTER_LINEAR)
            self.right_rect = cv2.remap(self.right_image_raw, calibration_configs.right_map1, calibration_configs.right_map2, cv2.INTER_LINEAR)

            cv2.imshow('left rect', self.left_rect)
            cv2.imshow('right rect', self.right_rect)
            cv2.waitKey(1)

            ### obtain the disp
            disparity = self.stereo.compute(self.left_rect, self.right_rect)
            disp_wls = wls_filter(self.stereo, self.left_rect, self.right_rect)
            self.disp = cv2.normalize(disp_wls, disp_wls, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # color depth
            # disp_color = disparity
            # disp_color = cv2.normalize(disp_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            # self.disp_color = cv2.applyColorMap(disp_color, 2)
            
            ### obtain the distance
            # Q = np.array([[1, 0  ,0     , -0.001 ],
            #             [0, 1  ,0     , -0.0526],
            #             [0, 0  ,0     , 0.0024 ],
            #             [0, 0  ,0.0015, 0      ]]) # obtain from cv2.stereoRectify

            # 3D coordiantes, where z is the distance
            threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32), calibration_configs.Q, handleMissingValues=True)
            threeD = threeD * 16
            # (1200+1000)/2, (200+500)/2
            x1, y1 = 1000, 850
            distance1 = math.sqrt(threeD[y1][x1][0]**2 + threeD[y1][x1][1]**2 + threeD[y1][x1][2]**2)
            distance1 = distance1 / 1000.0 # m
            ave_depth1.append(distance1)
            x2, y2 = 500, 700
            distance2 = math.sqrt(threeD[y2][x2][0]**2 + threeD[y2][x2][1]**2 + threeD[y2][x2][2]**2)
            distance2 = distance2 / 1000.0 # m
            ave_depth2.append(distance2)
            x3, y3 = 700, 550
            distance3 = math.sqrt(threeD[y3][x3][0]**2 + threeD[y3][x3][1]**2 + threeD[y3][x3][2]**2)
            distance3 = distance3 / 1000.0 # m
            ave_depth3.append(distance3)
            
            ### post precessing
            time2 = time.time()
            infer_time = time2 - time1
            self.disp = cv2.putText(self.disp, "Inference Time = %.2fs/frame" % (infer_time), org=(0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
            self.disp = cv2.rectangle(self.disp, (1000,1000), (1000,700), color=(0,0,0), thickness=2)
            self.disp = cv2.rectangle(self.disp, (600,600), (400,800), color=(0,0,0), thickness=2)
            self.disp_color = cv2.rectangle(self.disp, (800,400), (600,700), color=(0,0,0), thickness=2)
            # self.disp = cv2.fillConvexPoly(self.disp, np.int32(np.array([[1160, 460], [1200, 460], [1160, 600],[1200, 600]])), color=(255,255,0))
            # self.disp = cv2.fillConvexPoly(self.disp, np.int32(np.array([[560, 760], [600, 760], [560, 900],[600, 900]])), color=(255,255,0))
            # self.disp = cv2.fillConvexPoly(self.disp, np.int32(np.array([[760, 660], [800, 660], [760, 800],[800, 800]])), color=(255,255,0))
            
            if len(ave_depth1) % 10 and len(ave_depth2) % 10 and len(ave_depth3) % 10:
                ave_dis1 = sum(i for i in ave_depth1) / len(ave_depth1)
                ave_dis2 = sum(i for i in ave_depth2) / len(ave_depth2)
                ave_dis3 = sum(i for i in ave_depth3) / len(ave_depth3)
                # right
                self.disp = cv2.putText(self.disp, "Depth: %.2fm" % (ave_dis1), (x1-100-50, y1-150-30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=3)
                # left
                self.disp = cv2.putText(self.disp, "Depth: %.2fm" % (ave_dis2), (x2-100-20, y2-100-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=3)
                # middle
                self.disp = cv2.putText(self.disp, "Depth: %.2fm" % (ave_dis3), (x3-100-20, y3-150-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=3)
            
            cv2.setMouseCallback('disp', onmouse_pick_points, threeD) # self.onmouse_pick_points
            # ros_time = rclpy.clock.Clock().now().to_msg()
            # cv2.imshow('disp', self.disp)
            cv2.imshow('disp', self.disp)
            cv2.waitKey(1)
            

            # ROS msg processing
            disp_msg = self.bridge.cv2_to_imgmsg(np.array(self.disp), "8UC1")
            # disp_msg = self.bridge.cv2_to_imgmsg(np.array(self.disp_color), "8UC3")
            
            ros_time_msg = rclpy.clock.Clock().now().to_msg()
            disp_msg.header.frame_id = "left_camera" 
            disp_msg.header.stamp = ros_time_msg

            self.disp_publish.publish(disp_msg)
        
            

    def disp_sub_callback(self, disp_msg):
        self.get_logger().info(f'Received_disp_image: {self.i}') 
        self.get_logger().info(f'disp_timestamp: {disp_msg.header.stamp}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    disp_Publisher = stereo_disp_vpi()
    rclpy.spin(disp_Publisher)
    disp_Publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
