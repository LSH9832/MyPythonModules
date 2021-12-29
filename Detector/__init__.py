from time import time

class BasicDetector(object):

    class Colors:
        def __init__(self):
            hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                   '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
            self.palette = [self.hex2rgb('#' + c) for c in hex]
            self.n = len(self.palette)

        def __call__(self, i, bgr=False):
            c = self.palette[int(i) % self.n]
            return (c[2], c[1], c[0]) if bgr else c

        @staticmethod
        def hex2rgb(h):
            return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    import cv2
    __cv2 = cv2
    __colors = Colors()
    _class_names = []
    __last_time = time()

    def __init__(self, name_list):
        self._class_names = name_list

    def _xywh2xyxy(self, boxes, center=True):
        """Get coordinates (x0, y0, x1, y0) from model output (x, y, w, h)"""
        all_boxes = []
        for x, y, w, h in boxes:
            if center:
                x0, y0 = (x - 0.5 * w), (y - 0.5 * h)
                x1, y1 = (x + 0.5 * w), (y + 0.5 * h)
            else:
                x0, y0 = x, y
                x1, y1 = x + w, y + h
            all_boxes.append([x0, y0, x1, y1])
        return all_boxes

    def fps(self):
        now_time = time()
        this_fps = 1./(now_time - self.__last_time)
        self.__last_time = now_time
        return this_fps

    def _relate2abs(self, relative_boxes, width, height):
        relative_boxes = self._xywh2xyxy(relative_boxes)
        scale = [width, height, width, height]
        return [[relative * this_scale for relative, this_scale in zip(relative_box, scale)] for relative_box in relative_boxes]

    def __plot_one_box(self, x, im, color=(128, 128, 128), label=None, line_thickness=3):
        # Plots one bounding box on image 'im' using OpenCV
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
        tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness

        c1, c2 = (int(x[0]), int((x[1]))), (int(x[2]), int((x[3])))

        self.__cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=self.__cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = self.__cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], (c1[1] - t_size[1] - 3) if (c1[1] - t_size[1] - 3) > 0 else (c1[1] + t_size[1] + 3)
            self.__cv2.rectangle(
                img=im,
                pt1=c1,
                pt2=c2,
                color=color,
                thickness=-1,
                lineType=self.__cv2.LINE_AA
            )

            self.__cv2.putText(
                img=im,
                text=label,
                org=(c1[0], c1[1] - 2) if (c1[1] - t_size[1] - 3) > 0 else (c1[0], c2[1] - 2),
                fontFace=0,
                fontScale=tl / 3,
                color=[225, 255, 255],
                thickness=tf,
                lineType=self.__cv2.LINE_AA
            )

    def plot_result(self, img, bboxes, names, confs, draw_label=True):
        for xyxy, name, conf in zip(bboxes, names, confs):
            this_label = f'{self._class_names[name]}: {conf:0.2f}' if draw_label else None
            this_color = self.__colors(name, True)
            self.__plot_one_box(xyxy, img, this_color, this_label, 2)
