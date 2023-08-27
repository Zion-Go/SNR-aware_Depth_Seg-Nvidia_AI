import numpy as np
import cv2
import sys
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
from rclpy.executors import MultiThreadedExecutor
import threading
import calibration_configs

left_resize_list, right_resize_list = [], []

class ImageMerge(Node):
    def __init__(self):
        super().__init__("ImageMerge")

        self.bridge = CvBridge()
        self.i = 1
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_ALL,
                                depth=20,
                                durability=QoSDurabilityPolicy.VOLATILE)
        
        self.resizel_sub = self.create_subscription(Image, 'left/image_resize', self.merge_sub_callback, qos_profile=qos_policy)
        self.resizer_sub = self.create_subscription(Image, 'right/image_resize', self.merge_sub_callback, qos_profile=qos_policy)
        self.merge_publish = self.create_publisher(Image, 'left/image_merge', 20)
        self.timer = self.create_timer(0.3, self.merge_pub_callback)

    def merge_sub_callback(self, resizel_sub, resizer_sub):
        ros_time = rclpy.clock.Clock().now().to_msg()
        resizel_sub.header.stamp = ros_time
        resizer_sub.header.stamp = ros_time
        # print(imgl_rect_suber.header.stamp, imgr_rect_suber.header.stamp)

        if self.resizel_sub.header.frame_id == "left_camera":
            self.left_raw = self.bridge.imgmsg_to_cv2(resizel_sub, "bgr8")
            self.left_raw_list.append(self.left_raw)
            # print('left raw', len(self.left_raw_list))
            # print('left raw', type(self.left_raw))
        if self.resizer_sub.header.frame_id == "right_camera":
            self.right_raw = self.bridge.imgmsg_to_cv2(resizer_sub, "bgr8")
            self.right_raw_list.append(self.right_raw)
             

    def merge_pub_callback(self):
        
        height = 400
        width = 600
        # for n in range(len(image_list)):
        image_resize = cv2.resize(self.image, (width,height), interpolation= cv2.INTER_LINEAR)
            # cv2.imshow("Resize_600x400", image_resize)
            # cv2.waitKey(1)
        self.get_logger().info(f'Resized_raw_image-{image_resize.shape}: {self.i}')
        self.i += 1
        
        resize_msg = self.bridge.cv2_to_imgmsg(np.array(image_resize), 'bgr8')
        ros_time = rclpy.clock.Clock().now().to_msg()
        resize_msg.header.frame_id = "left_camera"
        resize_msg.header.stamp = ros_time

        self.resize_publish.publish(resize_msg)

        # del image_list[:n]

def main(args=None):
    rclpy.init(args=args)
    resize_publisher = ImageResize()
    rclpy.spin(resize_publisher)
    resize_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()