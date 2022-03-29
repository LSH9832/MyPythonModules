from yolox import Detector, draw_bb
import cv2

if __name__ == '__main__':
    detector = Detector(
        model_path="path to your yolox_s.pth",
        model_size="s",     # tiny, s, m, l
        class_path="path to your coco2017.txt",
        conf=0.25,
        nms=0.4,
        input_size=640,
        fp16=True
    )

    detector.loadModel()

    """
    or create detector as follows
    """
    # from yolox import create_detector_from_settings
    # detector = create_detector_from_settings("./demo/detect_settings.yaml")

    cap = cv2.VideoCapture("path to your test.mp4")
    while cap.isOpened():
        success, image = cap.read()
        if success:
            result = detector.predict(image)

            draw_bb(image, result, detector.get_all_classes())
            cv2.imshow("image", image)
            if cv2.waitKey(1) == 27:    # esc
                cv2.destroyAllWindows()
                break
