from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2
import torch
import numpy as np
from .pysot.core.config import cfg
from .pysot.models.model_builder import ModelBuilder
from .pysot.tracker.tracker_builder import build_tracker
torch.set_num_threads(1)

this_file_dir = str(__file__).replace('\\', '/').replace('__init__.py', '')


class Tracker(object):
    is_tracking = True

    def __init__(self, model_name='siamrpn_alex_dwxcorr_otb', is_tracking=True):
        self.is_tracking = is_tracking
        # load config
        config = this_file_dir + 'experiments/%s/config.yaml' % model_name
        snapshot = this_file_dir + 'experiments/%s/model.pth' % model_name
        cfg.merge_from_file(config)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        # create model
        self.model = ModelBuilder()
        # load model
        self.model.load_state_dict(torch.load(snapshot, map_location=lambda storage, loc: storage.cpu()))
        self.model.eval().to(device)

        # build tracker
        self.tracker = build_tracker(self.model)

        # self.model.half()
        self.set_boundingbox(np.zeros([3, 3, 3]), [0, 0, 1, 1])

    def set_boundingbox(self, image, boundingbox):
        """
        input:
            frame: bgr image
            bb: bounding box (x_left_top, y_left_top, w, h)

        """
        self.tracker.init(image, boundingbox)

    @staticmethod
    def draw_boundingbox(frame, result, color=(0, 255, 0), thickness=3, show_conf=True):
        frame = frame.copy()
        if 'polygon' in result:
            polygon = np.array(result['polygon']).astype(np.int32)
            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                          True, (0, 255, 0), 3)
            mask = ((result['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            mask = np.round(mask).astype(np.uint8)
            # print('mask', np.max(mask))
            mask = np.stack([0 * mask, mask, 0 * mask]).transpose([1, 2, 0])
            # cv2.imshow('mask',mask)
            mask_weight = 0.8

            background = frame * (1 - (mask == 255)).astype(np.uint8)
            this_object = frame * (mask == 255).astype(np.uint8)
            frame = background + cv2.addWeighted(this_object, 1. - mask_weight, mask, mask_weight, -1)
        else:
            bbox = list(map(int, result['bbox']))
            if show_conf:
                cv2.putText(frame, '%.3f' % result['best_score'], (bbox[0], bbox[1]+20), 0, 0.7, color, thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(
                img=frame,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                color=color,
                thickness=thickness
            )
        return frame

    def update(self, frame):
        outputs = self.tracker.track(frame)
        return outputs




# class TRTracker(object):
#
#     def __init__(self, model_name='siamrpn_alex_dwxcorr_otb'):
#         from torch2trt import TRTModule
#
#
#         # load config
#         config = this_file_dir + 'experiments/' + model_name + '/config.yaml'
#         ori_snapshot = this_file_dir + 'experiments/' + model_name + '/model.pth'
#         snapshot = this_file_dir + 'experiments/' + model_name + '/model_trt.pth'
#         cfg.merge_from_file(config)
#         cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
#         device = torch.device('cuda' if cfg.CUDA else 'cpu')
#
#         # create model
#         self.model = ModelBuilder()
#         # load model
#
#         self.model.load_state_dict(torch.load(ori_snapshot, map_location=lambda storage, loc: storage.cpu()))
#
#         ckpt = torch.load(snapshot, map_location=lambda storage, loc: storage.cpu())
#         for part in ckpt:
#             if ckpt[part]['success']:
#                 if part == "backbone":
#                     self.model.backbone = TRTModule()
#                     self.model.backbone.load_state_dict(ckpt[part]['weight'])
#                     self.model.backbone.cuda()
#                 elif part == "rpn_head":
#                     self.model.rpn_head = TRTModule()
#                     self.model.rpn_head.load_state_dict(ckpt[part]['weight'])
#                     self.model.rpn_head.cuda()
#
#         self.model.eval().to(device)
#
#         # build tracker
#         self.tracker = build_tracker(self.model)
#
#     def set_bb(self, frame, bb):
#         self.tracker.init(frame, bb)
#
#     def update(self, frame):
#         outputs = self.tracker.track(frame)
#         if 'polygon' in outputs:
#             polygon = np.array(outputs['polygon']).astype(np.int32)
#             cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
#                           True, (0, 255, 0), 3)
#             mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
#             mask = np.round(mask).astype(np.uint8)
#             # print('mask', np.max(mask))
#             mask = np.stack([0 * mask, mask, 0 * mask]).transpose(1, 2, 0)
#             # cv2.imshow('mask',mask)
#             mask_weight = 0.8
#
#
#
#             background = frame * (1 - (mask==255)).astype(np.uint8)
#             this_object = frame * (mask==255).astype(np.uint8)
#             frame = background + cv2.addWeighted(this_object, 1. - mask_weight, mask, mask_weight, -1)
#
#
#         else:
#             bbox = list(map(int, outputs['bbox']))
#             cv2.rectangle(frame, (bbox[0], bbox[1]),
#                           (bbox[0] + bbox[2], bbox[1] + bbox[3]),
#                           (0, 255, 0), 3)
#         return outputs, frame
#
#
# class Converter(object):
#
#     def __init__(self, model_name='siamrpn_alex_dwxcorr_otb'):
#         # load config
#         config = this_file_dir + 'experiments/' + model_name + '/config.yaml'
#         snapshot = this_file_dir + 'experiments/' + model_name + '/model.pth'
#         cfg.merge_from_file(config)
#         cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
#         device = torch.device('cuda' if cfg.CUDA else 'cpu')
#         # create model
#         self.weight_file = snapshot
#
#         self.model = ModelBuilder()
#         # load model
#         self.model.load_state_dict(torch.load(snapshot, map_location=lambda storage, loc: storage.cpu()))
#
#         self.model.eval().to(device)
#
#         # build tracker
#         self.tracker = build_tracker(self.model)
#
#
#         self.imgsz = 640
#         self.x = torch.zeros(1, 3, self.imgsz, self.imgsz).cuda()
#
#
#     def __convert_one_part(self, part, x, workspace, max_batch_size):
#         from torch2trt import torch2trt
#         import tensorrt as trt
#         part.cuda()
#         part.eval()
#
#         print(part)
#
#         try:
#             model_trt = torch2trt(
#                 part,
#                 [self.x],
#                 fp16_mode=True,
#                 log_level=trt.Logger.INFO,
#                 max_workspace_size=(1 << workspace),
#                 max_batch_size=max_batch_size,
#             )
#
#             success, weight = True, model_trt.state_dict()
#
#         except:
#             success, weight = False, part.state_dict()
#
#         return {'weight': weight, 'success': success}
#
#
#     def convert2trt(self, trt_filepath=None, workspace=32, max_batch_size=1):
#         from torch2trt import torch2trt
#         import tensorrt as trt
#         import os
#         if trt_filepath is None:
#             trt_filepath = '%s_trt.pth' % self.weight_file.split('.pth')[0]
#
#         data_to_save = dict()
#
#         # convert_model = [self.model.backbone, self.model.rpn_head]
#
#         data_to_save['backbone'] = self.__convert_one_part(
#             part=self.model.backbone,
#             x=self.x,
#             workspace=workspace,
#             max_batch_size=max_batch_size
#         )
#
#         self.x = self.model.backbone(self.x.cuda())
#
#         data_to_save['rpn_head'] = self.__convert_one_part(
#             part=self.model.rpn_head,
#             x=self.x,
#             workspace=workspace,
#             max_batch_size=max_batch_size
#         )
#
#
#         # print(data_to_save)
#         print('convertion finished, writing to pt file...')
#         torch.save(data_to_save, trt_filepath)
#         # torch.save(model_trt.state_dict(), trt_filepath)
#
#         print('done')
#
#
#
#         # print(self.model.backbone, self.model.rpn_head)

