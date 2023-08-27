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

# image_list = []

class Resize_Save_Sub(Node):
    def __init__(self):
        super().__init__("Resize_Save_Sub")

        self.bridge = CvBridge()
        self.i = 1
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_ALL,
                                depth=20,
                                durability=QoSDurabilityPolicy.VOLATILE)
        
        self.resize_sub = self.create_subscription(Image, 'left/image_resize', self.resize_sub_callback, qos_profile=qos_policy)
        # self.resize_publish = self.create_publisher(Image, 'left/image_resize', 20)
        # publish frequency -> model input? receive frequency?
        # self.timer = self.create_timer(2.0, self.resize_pub_callback)

    def resize_sub_callback(self, raw_msg):
        self.get_logger().info(f'Saved_resized_image: {self.i}')
        # self.get_logger().info(f'Resize_timestamp: {raw_msg.header.stamp}')
        self.image = self.bridge.imgmsg_to_cv2(raw_msg, 'bgr8')
        cv2.imwrite(f'{self.i}.png', self.image)
        self.i += 1
        # image_list.append(self.image)

    def resize_pub_callback(self):
        
        height = 400
        width = 600
        for n in range(len(image_list)):
            self.image_from_list = image_list[n]
            image_resize = cv2.resize(self.image_from_list, (width,height), interpolation= cv2.INTER_LINEAR)
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
    resize_save_sub = Resize_Save_Sub()
    rclpy.spin(resize_save_sub)
    resize_save_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()