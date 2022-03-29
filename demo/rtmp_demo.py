import sys

import cv2
import numpy as np

from yolox import multiDetectRtmpServer, draw_bb

sources = {
    "/home/uagv/Videos/test5.mp4": "rtmp://28.11.84.29/live/test1",
    "/home/uagv/Videos/test4.mp4": "rtmp://28.11.84.29/live/test2",
    # "/home/uagv/Videos/6.mp4": "rtmp://localhost/live/test3"
}


def track_object(data):
    from pysot import Tracker
    from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QSlider, QLabel
    from PyQt5.QtGui import QIcon, QPixmap, QImage, QPalette, QBrush
    from PyQt5.QtCore import QTimer
    from PyQt5.uic import loadUi

    class TrackingSystem(QMainWindow):

        def __init__(self, parent=None):
            super(QMainWindow, self).__init__(parent)
            loadUi("./tracking_system.ui", self)

            self.timer01 = QTimer()
            self.timer01.timeout.connect(self.timeoutFunction01)

            self.frame.mousePressEvent = self.shiftMode

            self.tracker = Tracker(is_tracking=False)
            self.index = 0

            self.setFixedSize(self.size())

            self.sources.clear()
            i = 0
            for _ in sources:
                i += 1
                self.sources.addItem("uav%d" % i)

            while not data["run"]:
                pass
            self.classes = data["classes"]
            self.timer01.start(30)

        def showImage(self, img, to):
            # to = self.frame if to is None else to
            if img is None:
                return False
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            to.setPixmap(QPixmap.fromImage(img))

        def timeoutFunction01(self):
            # a = QComboBox()
            # a = QSlider()
            # a = QLabel()
            # a.size().width()
            # a.value()


            conf, nms = float(self.conf_thres.value()) / 100.0, float(self.nms_thres.value()) / 100.0
            self.conf_show.setText("%.2f" % conf)
            self.nms_show.setText("%.2f" % nms)
            data["conf"] = conf
            data["nms"] = nms

            index = self.sources.currentIndex()
            if not self.index == index:
                self.tracker.is_tracking = False
                self.index = index
            i = 0
            this_source = None
            for source in sources:
                if i == index:
                    self.this_source = source
                    break
                i += 1
            if ("result_%s" % self.this_source) in data:
                self.this_result = data["result_%s" % self.this_source]
                this_image = data["img_%s" % self.this_source]
                self.ori_h, self.ori_w, _ = this_image.shape
                # print(self.frame.size()[0])
                this_image = cv2.resize(this_image, (self.frame.size().width(), self.frame.size().height()))
                if not self.tracker.is_tracking:
                    draw_bb(this_image, self.this_result, self.classes)

                elif self.tracker.is_tracking:
                    self.output = self.tracker.update(this_image)
                    this_image = self.tracker.draw_boundingbox(this_image, self.output, show_conf=False)
                self.showImage(this_image, self.frame)
            # print(index)

        def shiftMode(self, e):
            if not self.tracker.is_tracking:
                x = e.x()
                y = e.y()
                for *xyxy, _, _, _ in self.this_result:
                    if xyxy[0] <= x / self.frame.size().width() * self.ori_w <= xyxy[2] and \
                            xyxy[1] <= y / self.frame.size().height() * self.ori_h <= xyxy[3]:
                        self.tracker.set_boundingbox(
                            cv2.resize(data["img_%s" % self.this_source], (self.frame.size().width(), self.frame.size().height())),
                            np.array([xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]) *
                            np.array([self.frame.size().width() / self.ori_w, self.frame.size().height() / self.ori_h] * 2)
                        )
                        self.tracker.is_tracking = True
            else:
                self.tracker.is_tracking = False

        def closeEvent(self, e):
            data["run"] = False
            e.accept()

    app = QApplication(sys.argv)
    window = TrackingSystem()
    window.show()
    sys.exit(app.exec())


def main():
    server = multiDetectRtmpServer(bitrate=7*1024*1024, detector_setting_file_name="detect_settings.yaml")

    for source in sources:
        server.add_source(source=source, url=sources[source], fps=30)

    server.add_extra_process(func=track_object, args=())

    try:
        server.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
