import time
import numpy as np
import glob
import os
import cv2
#from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import CompressedImage, Image

RGB = 0
RGBD = 1
ROS = 2
FILE = 3
VIDEO = 0
VIDEO_NAME = 0

ROS_COMPRESSED = False
FILE_ADDRESS = '/imgs'
FILE_TYPE = 'jpg'
DEFAULT_SOURCE_TYPE = 0

this_dir = str(__file__).replace(str(__file__).split('/')[-1],'')

def setType(source_type):
    global DEFAULT_SOURCE_TYPE
    DEFAULT_SOURCE_TYPE = source_type


def getType():
    return DEFAULT_SOURCE_TYPE


def compressed_data2img(data):
    data = data.data
    data = np.asarray(bytearray(data))
    data = np.reshape(np.array(data), (np.shape(data)[0], 1))
    return cv2.imdecode(data.astype('uint8'), cv2.IMREAD_COLOR)


def data2img(data):
    imgdata = np.asarray(bytearray(data.data))
    img = np.reshape(imgdata, (data.height, data.width, 3))[:,:,::-1]  # reshape and rgb2bgr
    # print(np.shape(data))
    return img        


class Camera(object):

    def __init__(self, process_function=None, source_type=None):
        if source_type is None:
            self.source_type = getType()
        else:
            self.source_type = source_type
        self.camera = None
        self.img_type = None
        self.process = process_function
        self.start()
        
    def start(self):
        if self.source_type == RGB:
            self.camera = cv2.VideoCapture(VIDEO_NAME)

        elif self.source_type == RGBD:
            import pyrealsense2 as rs
            self.camera = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.camera.start(config)

        elif self.source_type == ROS:
            if ROS_COMPRESSED:
                self.topic_name = '/camera/color/image_raw/compressed'
                self.img_type = CompressedImage
            else:
                self.topic_name = '/camera/color/image_raw'
                self.img_type = Image

        elif self.source_type == FILE:
            self.camera = sorted(glob.glob(FILE_ADDRESS+'/*.'+FILE_TYPE))

    def get_img(self,data=None):
        # print(self.source_type)
        if self.source_type == RGB:
            if self.camera.isOpened():
                ret, frame = self.camera

        elif self.source_type == RGBD:
            # print('wait for frames')
            frames = self.camera.wait_for_frames()
            
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
            
        elif self.source_type == ROS:
            if ROS_COMPRESSED:
                frame = compressed_data2img(data)
            # print(type(np.asarray(bytearray(data.data))))
            else:
                frame = data2img(data)

        elif self.source_type == FILE:
            if len(self.camera):
                frame = cv2.imread(self.camera[0])
                # print(self.camera[0])
                del self.camera[0]
            else: frame = None
        return frame

    def callback(self, data):
        img = self.get_img(data)

        if img is None:
            if not self.source_type == ROS:
                return False
            else:
                # os.system('kill ' + self.rosimgpid)
                os.system('kill ' + str(os.getpid()))
        if not self.process == None:
            if not self.process(img):

                if self.source_type == ROS:
                    # os.system('kill ' + self.rosimgpid)
                    os.system('kill ' + str(os.getpid()))
                else:
                    return False

        if not self.source_type == ROS:
            return True

    def run(self):
        if self.source_type == ROS:
            import rospy
            
            rospy.init_node('webcam_display', anonymous=True)

            rospy.Subscriber(self.topic_name, self.img_type, self.callback, queue_size = 1, buff_size = 2764800)       # support max_buff_size=1280*720*3

            rospy.spin()
        else:
            while self.callback(None):
                pass

if __name__ == '__main__':
    pass
