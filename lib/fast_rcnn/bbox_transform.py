# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    ex_ctr_theta = ex_rois[:,4]

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    gt_ctr_theta = gt_rois[:,4] * 3.141592653/180

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets_theta = (gt_ctr_theta - ex_ctr_theta)
    #targets_theta = 0

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_theta)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths  = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x   = boxes[:, 0] + 0.5 * widths
    ctr_y   = boxes[:, 1] + 0.5 * heights
    theta   = boxes[:, 4]

    dx     = deltas[:, 0::5]
    dy     = deltas[:, 1::5]
    dw     = deltas[:, 2::5]
    dh     = deltas[:, 3::5]
    dtheta = deltas[:, 4::5]

    print 'widths',widths
    print 'widths_shape', widths.shape
    print 'widths[;,np.new]',widths[:, np.newaxis]
    print 'widths[;,np.new]_shape',widths[:, np.newaxis].shape
    print 'dx',dx
    print 'dx_shape',dx.shape

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w     = np.exp(dw) * widths[:, np.newaxis]
    pred_h     = np.exp(dh) * heights[:, np.newaxis]
    pred_theta = (theta[:,np.newaxis] + dtheta)

    print('theta_shape', theta.shape)
    print('theta', theta)
    print('dtheta_shape', dtheta.shape)
    print('dtheta', dtheta)

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::5] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::5] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::5] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::5] = pred_ctr_y + 0.5 * pred_h
    # theta
    pred_boxes[:, 4::5] = pred_theta

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::5] = np.maximum(np.minimum(boxes[:, 0::5], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::5] = np.maximum(np.minimum(boxes[:, 1::5], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::5] = np.maximum(np.minimum(boxes[:, 2::5], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::5] = np.maximum(np.minimum(boxes[:, 3::5], im_shape[0] - 1), 0)
    return boxes
