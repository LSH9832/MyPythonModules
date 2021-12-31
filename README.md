# MyPythonModules
自己积攒的Python功能性代码，封装成开源的第三方库以便使用，不断更新中。有一些模块只支持ubuntu18.04系统，甚至20.04都不一定能正常使用。**（可恶的cvbridge）**

## 1. Camera(旧) 和 camera（新）
### 1.1 Camera
这个包可以支持读取文件夹中的图片集、普通摄像头、realsense系列的RGBD摄像头（直接读取或从ros中订阅均可），**不保证在ubuntu18.04以外的系统上的可用性。**

#### 1.1.1 需要安装的第三方模块
- opencv-python或opencv-contrib-python
- rospy, 记得安装ROS相关的包如realsense_camera, realsense_rgbd等。
- numpy
- pyrealsense
- cvbridge3(安装不上就算了，只是极少部分功能不可用)

#### 1.1.2 用法示例
暂无
### 1.2 camera
目前支持图片集、普通摄像头和双目相机，暂不支持RGBD相机，**但是Windows和Ubuntu通用**
#### 1.2.1 需要安装的第三方模块
- opencv-python或opencv-contrib-python
#### 1.2.2 用法示例
```python3
import camera as cv2

"""
Other support format of source
    source = 0
    source = '/home/ubuntu/Videos/test.mp4'
    source = ['/home/ubuntu/Pictures/car6/*1.jpg', '/home/ubuntu/Pictures/car6/*6.jpg']
"""

source = '/home/ubuntu/Pictures/test/*.jpg'
double_eye = False

cam = cv2.OpenSource(
    source=source,
    double_eyes=double_eye      # 默认为 False, 不是双目相机时可不填
)
cam.setsize(1280, 720) if double_eye else None

cv2.fps(0)
while cam.isOpened():

    if double_eye:
        success, frame_left, frame_right = cam.readall()
        """
        success, frame_left = cam.readleft()
        success, frame_right = cam.readright()
        """
        frame = frame_left  # frame_right
    else:
        success, frame = cam.read()

    if success:
        else:
            cv2.fps(1)
            cv2.imshow('result', frame_show)
            fps2 = cv2.fps(1)
            if cv2.waitKey(1) == 27:
                break
            
            fps1 = cv2.fps(0)
            print('\rLoop FPS:%.2f  Draw FPS:%.2f' % (fps1, fps2), end='')


cv2.destroyAllWindows()
cam.release()
```


## 2. KCF
2014年发表的KCF跟踪算法，效果放到现在依然挺不错。opencv包中有，但貌似速度没有这个快，github上有源码，我封装了一下，更方便使用了。双系统均可用。

#### 2.1 需要安装的第三方模块
- opencv-python或opencv-contrib-python
- numpy
- numba numba版本一定要安装与你安装的numpy兼容的哦

#### 2.2 用法示例
```python3
from KCF import KCF
import cv2

tracker = KCF()
cam = cv2.VideoCapture(0)
x,y,w,h = tracker.choose_bb(source=cam, dynamic=False)
tracker.setbb((x,y,w,h))

time_delay = 1

while cam.isOpened():
    success, frame = cam.read()
    if success:
        frame, location = tracker.update(frame)
        print('\rlocation(x, y, w, h): %s' % location, end='')
        cv2.imshow('result', frame)

        press_key = cv2.waitKey(time_delay)
        if press_key == 27:
            cv2.destroyAllWindows()
            cam.release()
            break
        elif press_key == ord(' '):
            time_delay = 1 - time_delay
        elif press_key == ord('r'):
            x, y, w, h = tracker.choose_bb(source=cam)
            tracker.setbb((x, y, w, h))
        elif press_key == ord('d'):
            x, y, w, h = tracker.choose_bb(source=cam, dynamic=False)
            tracker.setbb((x, y, w, h))

```

## 3. 暂无
