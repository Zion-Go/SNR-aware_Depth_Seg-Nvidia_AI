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
import calibration_configs
import time

def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print(f'==============================\nPixel coordinates: x = {x}, y = {y}')
        # print("World coordinates：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("World coordinates：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0]**2 + threeD[y][x][1]**2 + threeD[y][x][2]**2)
        distance = distance / 1000.0  # mm -> m
        print("Distance：", distance, "m\n==============================")

def sad(imgL, imgR):
	
	return np.sum(np.abs(np.subtract(imgL, imgR, dtype=np.float)))

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

        # global img_rect_l, img_rect_r
        # global ros_time
        
        self.bridge = CvBridge()
        self.i = 1
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_ALL,
                                depth=20,
                                durability=QoSDurabilityPolicy.VOLATILE)
        # while True:
        #     try:
        # self.imgl_rect_sub = self.create_subscription(Image, 'left/image_rect', self.image_rect_l_callback, qos_profile=qos_policy)
        # self.imgr_rect_sub = self.create_subscription(Image, 'right/image_rect', self.image_rect_r_callback, qos_profile=qos_policy)
        self.imgl_raw_suber = message_filters.Subscriber(self, Image, "/left/image_raw", qos_profile=qos_policy)
        self.imgr_raw_suber = message_filters.Subscriber(self, Image, "/right/image_raw", qos_profile=qos_policy)
        
        sync_raw = message_filters.ApproximateTimeSynchronizer(
        [self.imgl_raw_suber, self.imgr_raw_suber],
        queue_size=200,
        slop=0.1,
        allow_headerless=False)
        sync_raw.registerCallback(self.sync_raw_callback)

        self.disp_publish = self.create_publisher(Image, '/disparity_stereo', 20)
        #### maybe problem
        self.timer = self.create_timer(0.2, self.disp_pub_callback)
        self.disp_sub = self.create_subscription(Image, '/disparity_stereo', self.disp_sub_callback, qos_profile=qos_policy)
            # except AttributeError:
            #     continue
        self.left_raw_list, self.right_raw_list = [], []

    # def onmouse_pick_points(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         threeD = param
    #         self.get_logger().info(f'===============\nPixel coordinates: x = {x}, y = {y}')
    #         # print("World coordinates：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
    #         self.get_logger().info("World coordinates：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

    #         distance = math.sqrt(threeD[y][x][0]**2 + threeD[y][x][1]**2 + threeD[y][x][2]**2)
    #         distance = distance / 1000.0  # mm -> m
    #         self.get_logger().info("Distance：", distance, "m\n===============")

    def sync_raw_callback(self, imgl_raw_suber, imgr_raw_suber):
        ros_time = rclpy.clock.Clock().now().to_msg()
        imgl_raw_suber.header.stamp = ros_time
        imgr_raw_suber.header.stamp = ros_time
        # print(imgl_rect_suber.header.stamp, imgr_rect_suber.header.stamp)

        # do mono preprocessing here
        if imgl_raw_suber.header.frame_id == "left_camera":
            self.left_raw = self.bridge.imgmsg_to_cv2(imgl_raw_suber, "mono8")
            self.left_raw_list.append(self.left_raw)
            print('left raw', len(self.left_raw_list))
            # print('left raw', type(self.left_raw))
        if imgr_raw_suber.header.frame_id == "right_camera":
            self.right_raw = self.bridge.imgmsg_to_cv2(imgr_raw_suber, "mono8")
            self.right_raw_list.append(self.right_raw)
            print('right raw', len(self.right_raw_list))
            # print('right raw', type(self.right_raw))
        
        # cv2.imshow('left raw', self.left_raw)
        # cv2.imshow('right raw', self.right_raw)
        # cv2.waitKey(1)
        

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

        # AttributeError: 'stereo_disp_vpi' object has no attribute 'left_raw'
        ### opencv remap stereorectify to get img rect
        for self.left_image_raw, self.right_image_raw in zip(self.left_raw_list, self.right_raw_list):
            

            self.left_rect = cv2.remap(self.left_image_raw, calibration_configs.left_map1, calibration_configs.left_map2, cv2.INTER_LINEAR)
            self.right_rect = cv2.remap(self.right_image_raw, calibration_configs.right_map1, calibration_configs.right_map2, cv2.INTER_LINEAR)

            cv2.imshow('left rect', self.left_rect)
            cv2.imshow('right rect', self.right_rect)
            cv2.waitKey(1)

            ### sift (version problem) to get img rect
            # sift = cv2.xfeatures2d.SIFT_create()
            # kp = sift.detect(self.left_rect, None)
            # left_sift = cv2.drawKeypoints(self.left_rect, kp, left_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow('left sift', left_sift)
            # cv2.waitKey(1)
            
            ### ORB to get img rect
            # orb = cv2.ORB_create()
            # kp1, des1 = orb.detectAndCompute(self.left_raw, None)
            # kp2, des2 = orb.detectAndCompute(self.right_raw, None)

            # bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
            # matches = bf.match(des1, des2)
            # matches = sorted(matches, key = lambda x:x.distance)[:30]

            # img_match = cv2.drawMatches(self.left_raw, kp1, self.right_raw, kp2, matches, flags=2, outImg = None)

            # best_kp1 = []
            # best_kp2 = []
            # best_matches = []

            # for match in matches:
            #     best_kp1.append(kp1[match.queryIdx].pt)
            #     best_kp2.append(kp2[match.trainIdx].pt)
            #     best_matches.append(match)

            # best_kp1 = np.array(best_kp1)
            # best_kp2 = np.array(best_kp2)
            # best_matches = np.array(best_matches)

            # F, inlier_mask = cv2.findFundamentalMat(best_kp1, best_kp2, cv2.FM_7POINT)
            # inlier_mask = inlier_mask.flatten()

            # #points within epipolar lines
            # inlier_kp1 = best_kp1[inlier_mask == 1]
            # inlier_kp2 = best_kp2[inlier_mask == 1]

            # inlier_matches = best_matches[inlier_mask==1]


            # img_match = cv2.drawMatches(self.left_raw, kp1, self.right_raw, kp2, inlier_matches, flags=2, outImg = None)
            # cv2.imshow('stereo matching images', img_match)
            # cv2.waitKey(1)

            # thresh = 0

            # _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(inlier_kp1), np.float32(inlier_kp2), F, self.left_raw.shape[::-1], 1)

            # l_rect = np.float32([[[0, 0], [self.left_raw.shape[1], 0], [self.left_raw.shape[1], self.left_raw.shape[0]], [0, self.left_raw.shape[0]]]])
            # warped_l_rect = cv2.perspectiveTransform(l_rect, H1)

            # r_rect = np.float32([[[0, 0], [self.right_raw.shape[1], 0], [self.right_raw.shape[1], self.right_raw.shape[0]], [0, self.right_raw.shape[0]]]])
            # warped_r_rect = cv2.perspectiveTransform(r_rect, H2)

            # min_x_l = min(warped_l_rect[0][0][0], warped_l_rect[0][1][0], warped_l_rect[0][2][0], warped_l_rect[0][3][0])
            # min_x_r = min(warped_r_rect[0][0][0], warped_r_rect[0][1][0], warped_r_rect[0][2][0], warped_r_rect[0][3][0])

            # min_y_l = min(warped_l_rect[0][0][1], warped_l_rect[0][1][1], warped_l_rect[0][2][1], warped_l_rect[0][3][1])
            # min_y_r = min(warped_r_rect[0][0][1], warped_r_rect[0][1][1], warped_r_rect[0][2][1], warped_r_rect[0][3][1])

            # max_x_l = max(warped_l_rect[0][0][0], warped_l_rect[0][1][0], warped_l_rect[0][2][0], warped_l_rect[0][3][0])
            # max_x_r = max(warped_r_rect[0][0][0], warped_r_rect[0][1][0], warped_r_rect[0][2][0], warped_r_rect[0][3][0])

            # max_y_l = max(warped_l_rect[0][0][1], warped_l_rect[0][1][1], warped_l_rect[0][2][1], warped_l_rect[0][3][1])
            # max_y_r = max(warped_r_rect[0][0][1], warped_r_rect[0][1][1], warped_r_rect[0][2][1], warped_r_rect[0][3][1])
            
            # translation_xy_l = np.array([max(0, -min_x_l), max(0, -min_y_l)])
            # translation_xy_r = np.array([max(0, -min_x_r), max(0, -min_y_r)])

            # W_l = (max_x_l + translation_xy_l[0])
            # H_l = (max_y_l + translation_xy_l[1])

            # W_r = (max_x_r + translation_xy_r[0])
            # H_r = (max_y_r + translation_xy_r[1])

            # transform_T = np.eye(3)
            # transform_T[0,2] = translation_xy_l[0]
            # transform_T[1,2] = translation_xy_l[1]
            # transform_T = transform_T[:2, :]

            # H1 = np.concatenate((transform_T, [[0, 0, 1]]), axis=0) @ H1

            # transform_T = np.eye(3)
            # transform_T[0,2] = translation_xy_r[0]
            # transform_T[1,2] = translation_xy_r[1]
            # transform_T = transform_T[:2, :]

            # H2 = np.concatenate((transform_T, [[0, 0, 1]]), axis=0) @ H2

            # self.left_rect = cv2.warpPerspective(self.left_raw, H1, (int(W_l),int(H_l)))
            # self.right_rect = cv2.warpPerspective(self.right_raw, H2, (int(W_r),int(H_r)))
            # print(self.left_rect.size)
            # print(self.left_rect.depth)
            # cv2.imshow('left rect', self.left_rect)
            # cv2.imshow('right rect', self.right_rect)
            # cv2.waitKey(1)

            ### obtain the disp
            disparity = self.stereo.compute(self.left_rect, self.right_rect)
            # sad_error = sad(self.left_rect, self.right_rect)
            disp_wls = wls_filter(self.stereo, self.left_rect, self.right_rect)
            self.disp = cv2.normalize(disp_wls, disp_wls, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # color depth
            # disp_color = disparity
            # disp_color = cv2.normalize(disp_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            # disp_color = cv2.applyColorMap(disp_color, 2)
            
            ### obtain the distance
            # Q = np.array([[1, 0  ,0     , -0.001 ],
            #             [0, 1  ,0     , -0.0526],
            #             [0, 0  ,0     , 0.0024 ],
            #             [0, 0  ,0.0015, 0      ]]) # obtain from cv2.stereoRectify

            # 3D coordiantes, where z is the distance
            threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32), calibration_configs.Q, handleMissingValues=True)
            threeD = threeD * 16
            
            x1, y1 = 980, 480
            distance1 = math.sqrt(threeD[y1][x1][0]**2 + threeD[y1][x1][1]**2 + threeD[y1][x1][2]**2)
            distance1 = distance1 / 1000.0 # m
            ave_depth1.append(distance1)
            x2, y2 = 380, 780
            distance2 = math.sqrt(threeD[y2][x2][0]**2 + threeD[y2][x2][1]**2 + threeD[y2][x2][2]**2)
            distance2 = distance2 / 1000.0 # m
            ave_depth2.append(distance2)
            x3, y3 = 580, 680
            distance3 = math.sqrt(threeD[y3][x3][0]**2 + threeD[y3][x3][1]**2 + threeD[y3][x3][2]**2)
            distance3 = distance3 / 1000.0 # m
            ave_depth3.append(distance3)
            
            ### post precessing
            time2 = time.time()
            infer_time = time2 - time1
            self.disp = cv2.putText(self.disp, "Inference Time = %.2fs/frame" % (infer_time), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(255, 255, 0), thickness=2)
            self.disp = cv2.rectangle(self.disp, (1200,200), (1000,500), (255, 0, 0), thickness=2)
            self.disp = cv2.rectangle(self.disp, (600,600), (400,800), (255, 0, 0), thickness=2)
            self.disp = cv2.rectangle(self.disp, (800,400), (600,700), (255, 0, 0), thickness=2)
            
            # while len(ave_depth) > 0:
            if len(ave_depth1) % 10 and len(ave_depth2) % 10 and len(ave_depth3) % 10:
                ave_dis1 = sum(i for i in ave_depth1) / len(ave_depth1)
                ave_dis2 = sum(i for i in ave_depth2) / len(ave_depth2)
                ave_dis3 = sum(i for i in ave_depth3) / len(ave_depth3)
                self.disp = cv2.putText(self.disp, "Depth: %.2fm" % (ave_dis1), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=1)
                self.disp = cv2.putText(self.disp, "Depth: %.2fm" % (ave_dis2), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=1)
                self.disp = cv2.putText(self.disp, "Depth: %.2fm" % (ave_dis3), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=1)
            
            cv2.setMouseCallback('disp', onmouse_pick_points, threeD) # self.onmouse_pick_points
            # ros_time = rclpy.clock.Clock().now().to_msg()
            cv2.imshow('disp', self.disp)
            # cv2.imshow('color disp', self.disp_color)
            cv2.waitKey(1)
            # print(sad_error)
            # print('disp', type(self.disp))

            # ROS msg processing
            disp_msg = self.bridge.cv2_to_imgmsg(np.array(self.disp), "8UC1")
            # disp_msg.f = 2364.070068359375
            # disp_msg.T = 0.0
            # disp_msg.min_disparity = -1
            # disp_msg.max_disparity = 16.0
            # disp_msg.delta_d = 0.0625
            
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