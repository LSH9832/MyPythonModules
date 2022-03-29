from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
os.chdir('..')
print(torch.cuda.is_available())
print(cv2.__version__)
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    # print(not video_name)
    if video_name:
        cap = cv2.VideoCapture(0)
        print(cap.isOpened())
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                pass
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    config = 'experiments/siamrpn_alex_dwxcorr_otb/config.yaml'
    snapshot = 'experiments/siamrpn_alex_dwxcorr_otb/model.pth'
    video_name = False
    cfg.merge_from_file(config)
    print(1)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    print(2)
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    print(3)
    # create model
    model = ModelBuilder()
    print(4)
    # load model
    model.load_state_dict(torch.load(snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    print(5)
    model.eval().to(device)
    print(6)
    # build tracker
    tracker = build_tracker(model)
    print(7)
    first_frame = True
    if video_name:
        video_name = video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    print(8)
    for frame in get_frames(video_name):
        if first_frame:
            init_rect = cv2.selectROI(video_name, frame, False, False)

            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            if cv2.waitKey(1)==27:
                break


if __name__ == '__main__':
    main()
# python tools/demo.py --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth