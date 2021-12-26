#! /usr/bin/python
import rospy
import pyrealsense2 as rs
from std_msgs.msg import UInt8MultiArray
import numpy as np
import cv2
import os

RGB = 0
RGB_ID = 0
RGBD = 1
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
this_dir = str(__file__).replace(str(__file__).split('/')[-1],'')


def img2data(img):
    img_encode = cv2.imencode('.jpg', img, encode_param)[1]
    data = UInt8MultiArray(data=np.reshape(img_encode, (np.shape(img_encode))[0], ).tolist())
    # print(max(img_encode))
    return data


def data2img(data):
    data = np.reshape(np.array(data), (np.shape(data)[0], 1))
    return cv2.imdecode(data.astype('uint8'), cv2.IMREAD_COLOR)


class ImgPublisher:

    def __init__(self, topic_name = 'ros_image', source = RGBD):
        self.__pub_ = rospy.Publisher(topic_name, UInt8MultiArray, queue_size = 50)
        self.camera = None
        self.source = source

        if self.source==RGBD:
            self.camera = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.camera.start(config)
        elif self.source==RGB:
            self.camera = cv2.VideoCapture(RGB_ID)

    def get_img(self):
        # print(self.source==RGBD)
        if self.source==RGBD:
            frames = self.camera.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
            return frame
 
        elif self.source==RGB:
            ret, frame = self.camera.read()
            if ret:
                return frame


    def pub(self):
        
        frame = self.get_img()
        str_send = img2data(frame)
        self.__pub_.publish(str_send)
    
    def run(self):
        while not rospy.is_shutdown():
            self.pub()
        
if __name__ == "__main__":
    rospy.init_node('ros_camera', anonymous=True)
    IMGPObject = ImgPublisher()
    IMGPObject.run()













