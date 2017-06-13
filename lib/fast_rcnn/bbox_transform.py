# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

DEBUG = False
def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2]
    ex_heights = ex_rois[:, 3] 
    ex_ctr_x = ex_rois[:, 0]
    ex_ctr_y = ex_rois[:, 1]
    ex_theta = ex_rois[:,4]

    gt_widths = gt_rois[:, 2]
    gt_heights = gt_rois[:, 3]
    gt_ctr_x = gt_rois[:, 0]
    gt_ctr_y = gt_rois[:, 1]
    gt_theta = gt_rois[:, 4]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    
    targets_theta  = gt_theta - ex_theta/3.1415926535*180
  
    if DEBUG:
        print ex_rois 
        print gt_rois 
        print('gt_theta')
        print(gt_theta)
        print('ex_theta')
        print(ex_theta)
        print ('targets_theta')
        print(targets_theta)

    for i in range(targets_theta.shape[0]):
        if(targets_theta[i] < -45):
            while (targets_theta[i] < -45):
                targets_theta[i] += 180
        elif(targets_theta[i] > 135):
            while(targets_theta[i] > 135):
                targets_theta[i] -= 180
        targets_theta[i] = targets_theta[i]/180 * 3.1415926535

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_theta)).transpose()
    return targets

def orient_bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2]
    heights = boxes[:, 3] 
    ctr_x = boxes[:, 0]
    ctr_y = boxes[:, 1] 

    dx = deltas[:, 0::5]
    dy = deltas[:, 1::5]
    dw = deltas[:, 2::5]
    dh = deltas[:, 3::5]

    if DEBUG:
        print 'ssss',deltas.shape
        print dx.shape,dy.shape,dw.shape,dh.shape
    
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    if DEBUG:
        print 'pred_ctr_shape',pred_ctr_x.shape
        print 'ctr_shape',ctr_x.shape
        print 'pred_w_shape',pred_w.shape

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::5] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::5] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::5] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::5] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]


    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
