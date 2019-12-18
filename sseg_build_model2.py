import sseg_build_graph
import sseg_loss
import sseg_data_input2 as sseg_data_input
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import common_utils as util
import torch
import torch.nn as nn
import torch.nn.functional as F
from sseg_config import Config
from ssd import SSD,build_ssd
import torchvision.transforms.Lambda as L
class SSDSEG(nn.module):

    def __init__(self, mode, config):
        super(SSDSEG,self).__init__()
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config

        # class InferenceConfig(Config):
        #     BATCH_SIZE = 4
        #self.test_config = Config()
        self.ssd = build_ssd(0,300,21)

    def forward(self,config,input_image):
        input_image = input_image[0]  # to get input read image input = [input_image,input_gt_boxc,input_gt_mask,input_face3_gt]
        #self.ssd(input_image)
        # build ssd_det
        # return det_rois: [batch_size, top_k, (class_id,score,xmin,ymin,xmax,ymax)]
        '''det_rois,boxc_pred,share_feature,face_mask = sseg_net.snet(input_image=input_image,
                                                                   image_size=(config.img_height, config.img_width, config.img_channels),
                                                                   n_classes=config.n_classes,
                                                                   l2_regularization=0.0005,
                                                                   scales=config.scales,
                                                                   aspect_ratios_per_layer=config.aspect_ratios,
                                                                   two_boxes_for_ar1=config.two_boxes_for_ar1,
                                                                   steps=config.steps,
                                                                   offsets=config.offsets,
                                                                   limit_boxes=config.limit_boxes,
                                                                   variances=config.variances,
                                                                   coords=config.coords,
                                                                   normalize_coords=config.normalize_coords,
                                                                   subtract_mean=config.subtract_mean,
                                                                   divide_by_stddev=None,
                                                                   swap_channels=config.swap_channels,
                                                                   confidence_thresh=0.7,
                                                                   iou_threshold=0.5,
                                                                   top_k=config.TOPK,
                                                                   nms_max_output_size=100,
                                                                   min_scale = None,
                                                                   max_scale = None,
'''
        det_rois, boxc_pred, share_feature, face_mask = self.ssd(input_image)

        Submask_ebs = sseg_build_graph.SubMask_eb(config=config,numclass=3)
        pred_mask1 = Submask_ebs([share_feature[0],share_feature[1],share_feature[2],rois1])
        pred_mask2 = Submask_ebs([share_feature[0],share_feature[1],share_feature[2],rois2])
        pred_mask3 = sseg_build_graph.submask_nosenet(input_feature_map=share_feature,numclass=2,config=config,rois=rois3)
        pred_mask4 = sseg_build_graph.submask_mousenet(input_feature_map=share_feature,numclass=4,config=config,rois=rois4)

        outputs = [pred_mask1,pred_mask2,pred_mask3,pred_mask4,face_mask,boxc_pred]
#        inputs = [input_image,input_gt_boxc,input_gt_mask,input_face3_gt]
        #outputs = [  boxc_loss ,mask_loss1,mask_loss2,mask_loss3,mask_loss4,mask_loss_face,
        #             target_mask3,pred_mask3]

        return outputs





