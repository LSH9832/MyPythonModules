import numpy as np
import cv2
# import sys
from . import kcftracker
# import multiprocessing as mp


class KCF(object):

    def __init__(self, q):
        self.q = q
        self.selectingObject = False
        self.initTracking = False
        self.onTracking = False
        self.ix, self.iy, self.cx, self.cy = -1, -1, -1, -1
        self.w, self.h = 0, 0
        self.cap = cv2.VideoCapture()
        self.tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale

        self.inteval = 1
        self.duration = 0.01

        self.tracker.init([1, 1, 2, 2], np.zeros([5,5,3]))


    def draw_boundingbox(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.selectingObject = True
            self.onTracking = False
            self.ix, self.iy = x, y
            self.cx, self.cy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            self.cx, self.cy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.selectingObject = False
            if abs(x - self.ix) > 10 and abs(y - self.iy) > 10:
                self.w, self.h = abs(x - self.ix), abs(y - self.iy)
                self.ix, self.iy = min(x, self.ix), min(y, self.iy)
                self.initTracking = True
            else:
                self.onTracking = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.onTracking = False
            if self.w > 0:
                self.ix, self.iy = x - self.w / 2, y - self.h / 2
                self.initTracking = True

    def run_camera(self, address = 0):
        self.cap = cv2.VideoCapture(address)

        cv2.namedWindow('tracking')
        cv2.setMouseCallback('tracking', self.draw_boundingbox)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, bb = self.update(frame)

            cv2.imshow('tracking', frame)
            c = cv2.waitKey(self.inteval) & 0xFF
            if c == 27 or c == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update(self, frame):
        boundingbox = None
        if self.selectingObject:
            cv2.rectangle(frame, (self.ix, self.iy), (self.cx, self.cy), (0, 255, 255), 1)
        elif self.initTracking:
            cv2.rectangle(frame, (self.ix, self.iy), (self.ix + self.w, self.iy + self.h), (0, 255, 255), 2)
            print([self.ix, self.iy, self.w, self.h])
            self.tracker.init([self.ix, self.iy, self.w, self.h], frame)

            self.initTracking = False
            self.onTracking = True

        elif self.onTracking:
            boundingbox = self.tracker.update(frame)
            boundingbox = list(map(int, boundingbox))
            # print(boundingbox)
            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]),
                          (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)

            # cv2.putText(frame, 'Tracking', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #             (0, 0, 255), 2)

        return frame, boundingbox

    def setbb(self, bb:tuple):
        # bb = (x,y,w,h)
        self.ix, self.iy, self.w, self.h = bb

        self.initTracking = True
        self.onTracking = False
        self.selectingObject = False

    def stop_tracking(self):
        self.selectingObject = False
        self.initTracking = False
        self.onTracking = False


    def choose_bb(self, address = 0):
        self.cap = cv2.VideoCapture(address)

        cv2.namedWindow('choosing bounding box')
        cv2.setMouseCallback('choosing bounding box', self.draw_boundingbox)

        while self.cap.isOpened() and not self.onTracking:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, bb = self.update(frame)

            cv2.imshow('choosing bounding box', frame)
            c = cv2.waitKey(self.inteval) & 0xFF
            if c == 27 or c == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.ix, self.iy, self.w, self.h

def demo():
    a = KCF(0)
    # a.setbb((277, 123, 159, 77))
    a.run_camera()

# 两矩形框重合度
XYWH = 1
XYXY = 0
def overlap(loc1:tuple, loc2:tuple, datatype = XYWH):
    x1,y1,w1,h1 = loc1
    x2,y2,w2,h2 = loc2
    left_top = (max(x1,x2),max(y1,y2))
    if datatype:
        right_down = (min(x1+w1,x2+w2),min(y1+h1,y2+h2))
    else:
        right_down = (min(w1, w2), min(h1, h2))

    s_and = 0
    if right_down[0] > left_top[0] and right_down[1] > left_top[1]:
        s_and = (right_down[0] - left_top[0]) * (right_down[1] - left_top[1])
    s_or = w1 * h1 + w2 * h2 - s_and
    return float(s_and) / float(s_or)




if __name__ == '__main__':
    # print(overlap((1,1,2,2),(0,0,1,1)))
    demo()
