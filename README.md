# MyPythonModules
自己积攒的Python功能性代码，封装成开源的第三方库以便使用，有一些模块只支持ubuntu18.04系统，甚至20.04都不一定能正常使用。**（可恶的cvbridge）**

## 1. Camera
这个包可以支持读取文件夹中的图片集、普通摄像头、realsense系列的RGBD摄像头（直接读取或从ros中订阅均可），**不保证在ubuntu18.04以外的系统上的可用性。**

#### 1.1 需要安装的第三方模块
- opencv-python或opencv-contrib-python
- rospy, 记得安装ROS相关的包如realsense_camera, realsense_rgbd等。
- numpy
- pyrealsense
- cvbridge3(安装不上就算了，只是极少部分功能不可用)

#### 1.2 用法示例
暂无

## 2. KCF
2014年发表的KCF跟踪算法，效果放到现在依然挺不错。opencv包中有，但貌似速度没有这个快，github上有源码，我封装了一下，更方便使用了。

#### 2.1 需要安装的第三方模块
- opencv-python或opencv-contrib-python
- numpy
- numba numba版本一定要安装与你安装的numpy兼容的哦

## 2.2 用法示例
暂无

# 3. 暂无
