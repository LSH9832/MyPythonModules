from __future__ import absolute_import


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
# from glob import glob
import Camera

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

print(torch.cuda.is_available())
print(cv2.__version__)
torch.set_num_threads(1)

# load config
config = 'experiments/siamrpn_alex_dwxcorr_otb/config.yaml'
snapshot = 'experiments/siamrpn_alex_dwxcorr_otb/model.pth'
video_name = False
cfg.merge_from_file(config)
cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
device = torch.device('cuda' if cfg.CUDA else 'cpu')
# create model
model = ModelBuilder()
# load model
model.load_state_dict(torch.load(snapshot,
    map_location=lambda storage, loc: storage.cpu()))
model.eval().to(device)
# build tracker



tracker = build_tracker(model)

if video_name:
    video_name = video_name.split('/')[-1].split('.')[0]
else:
    video_name = 'webcam'
cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)


first_frame = True
init_ok = False
def this_callback(frame):

    global first_frame, init_ok
    if first_frame:
        first_frame = False
        init_rect = cv2.selectROI(video_name, frame, False, False)
        tracker.init(frame, init_rect)
        init_ok = True
    elif init_ok:
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
            return False
    return True

if __name__ == '__main__':
    Camera.FILE_ADDRESS = '/media/shl/数据资料文档/UAV123/data_seq/UAV123/person4'
    cam = Camera.Camera(Camera.FILE,this_callback)
    cam.run()
# python tools/demo.py --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth
