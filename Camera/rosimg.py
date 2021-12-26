#! /usr/bin/python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32MultiArray
import numpy as np
import cv2
import os
import zlib

topic_name = '/camera/color/image_raw/compressed'
bridge = CvBridge()
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 99]

def img2data(img):
    img_encode = cv2.imencode('.jpg', img, encode_param)[1]
    
    # print('data: ', np.reshape(img_encode, (np.shape(img_encode))[0], ))
    data = Int32MultiArray(data=np.reshape(img_encode, (np.shape(img_encode))[0], ).tolist())
    return data

def data2img(data):
    data = np.reshape(np.array(data), (np.shape(data)[0], 1))
    return cv2.imdecode(data.astype('uint8'), cv2.IMREAD_COLOR)


class SubscribeAndPublish:
    def __init__(self):
        self.__pub_ = rospy.Publisher('py2py3', Int32MultiArray, queue_size = 50)
        self.__sub_ = rospy.Subscriber(topic_name, CompressedImage, self.callback)

	self.first_frame = True
        # self.__rec_ = rospy.Subscriber('/py2py3', Int32MultiArray, self.receive_callback)

    def callback(self,data):
        # assert isinstance(data, Sub msg Type) 
        frame = bridge.compressed_imgmsg_to_cv2(data, "bgr8")

        if self.first_frame:
            self.first_frame = False
            pid = Int32MultiArray(data = [int(os.getpid())])
            self.__pub_.publish(pid)
        
        str_send = img2data(frame)
        
        self.__pub_.publish(str_send)
        
    
    def receive_callback(self,data):
        data = data.data
        
        frame = data2img(data)
        cv2.imshow('test',frame)
        if cv2.waitKey(1)==27:
            os.system('kill ' + str(os.getpid()))

def main():
    rospy.init_node('webcam_display', anonymous=True)
    SAPObject = SubscribeAndPublish()
    rospy.spin()

if __name__ == "__main__":
    main()

