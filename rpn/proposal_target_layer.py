# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import rect_bbox_transform
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import time

# DEBUG = False
DEBUG = True 

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 6)

        # rpn_box_rois
        top[1].reshape(1, 6)
        # labels
        top[2].reshape(1, 1)
        # bbox_targets
        top[3].reshape(1, self._num_classes * 5)
        # bbox_inside_weights
        top[4].reshape(1, self._num_classes * 5)
        # bbox_outside_weights
        top[5].reshape(1, self._num_classes * 5)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rect_rois = bottom[0].data

        all_box_rois = bottom[1].data

        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[2].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)

        overlap_gt_boxes = boxes_transform_to_rect(gt_boxes)

        # print all_rect_rois[0]
        # print overlap_gt_boxes[0]
        # time.sleep(10)
        all_rois = np.vstack(
            (all_rect_rois, np.hstack((zeros, overlap_gt_boxes[:, :-1])))
        )

        all_box_rois = np.vstack((all_box_rois, np.hstack((zeros, gt_boxes[:,:-1]))))

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rect_rois, box_rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, all_box_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        if DEBUG:
            fg_num = 0
            bg_num = 0
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            fg_num += (labels > 0).sum()
            bg_num += (labels == 0).sum()
            # print 'num fg avg: {}'.format(fg_num / count)
            # print 'num bg avg: {}'.format(bg_num / count)
            print 'ratio: {:.3f}'.format(float(fg_num) / float(bg_num))

        # time.sleep(10)
        # sampled rois
        top[0].reshape(*rect_rois.shape)
        top[0].data[...] = rect_rois

        # sampled rois
        top[1].reshape(*box_rois.shape)
        top[1].data[...] = box_rois

        # classification labels
        top[2].reshape(*labels.shape)
        top[2].data[...] = labels

        # bbox_targets
        top[3].reshape(*bbox_targets.shape)
        top[3].data[...] = bbox_targets

        # bbox_inside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[5].reshape(*bbox_inside_weights.shape)
        top[5].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 5 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 5 * cls
        end = start + 5
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
        # bbox_inside_weights[ind, start:end] = cfg.TRAIN.RECT_BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 5
    assert gt_rois.shape[1] == 5

    targets = bbox_transform(ex_rois, gt_rois)
    # targets = rect_bbox_transform(ex_rois, gt_rois)
    # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # # Optionally normalize targets by a precomputed mean and stdev
        # targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                # / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rect_rois, all_box_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)

    # overlap_gt_boxes = gt_boxes[:,:5]
    overlap_rois     = all_rect_rois[:,1:6]
    overlap_gt_boxes = gt_boxes[:,:5]
    
    # overlap_rois = boxes_transform_to_rect(overlap_rois)
    overlap_gt_boxes = boxes_transform_to_rect(overlap_gt_boxes)

    # print 'overlap_rois',overlap_rois
    # print 'overlap_gt_boxes',overlap_gt_boxes

    overlaps = bbox_overlaps(
        np.ascontiguousarray(overlap_rois[:, :4], dtype=np.float),
        np.ascontiguousarray(overlap_gt_boxes[:, :4], dtype=np.float))

    # print 'overlaps', overlaps

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 5]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0

    rect_rois = all_rect_rois[keep_inds]
    box_rois = all_box_rois[keep_inds]
    
    # print box_rois[0]
    # print rect_rois[0]

    #rewrite compute targets function
    # print labels
    bbox_target_data = _compute_targets(
        box_rois[:, 1:6], gt_boxes[gt_assignment[keep_inds], :5], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rect_rois, box_rois, bbox_targets, bbox_inside_weights

def boxes_transform_to_rect(gt_boxes):
    overlap_gt_boxes = gt_boxes.copy()

    angle = gt_boxes[:,4]
    a = np.cos(angle)/2
    b = np.sin(angle)/2

    p_0_x = gt_boxes[:,0] - gt_boxes[:,2] * a + gt_boxes[:,3] * b
    p_0_y = gt_boxes[:,1] - gt_boxes[:,2] * b - gt_boxes[:,3] * a
    p_3_x = gt_boxes[:,0] - gt_boxes[:,2] * a - gt_boxes[:,3] * b
    p_3_y = gt_boxes[:,1] - gt_boxes[:,2] * b + gt_boxes[:,3] * a

    p_1_x = 2 * gt_boxes[:,0] - p_3_x
    p_1_y = 2 * gt_boxes[:,1] - p_3_y
    p_2_x = 2 * gt_boxes[:,0] - p_0_x
    p_2_y = 2 * gt_boxes[:,1] - p_0_y

    x_min = []
    y_min = []
    x_max = []
    y_max = []

    for i in range(0,len(p_0_x)):
        x_min.append(max(0,min(p_0_x[i],p_1_x[i],p_2_x[i],p_3_x[i])))
        x_max.append(max(p_0_x[i],p_1_x[i],p_2_x[i],p_3_x[i]))
        y_min.append(max(0,min(p_0_y[i],p_1_y[i],p_2_y[i],p_3_y[i])))
        y_max.append(max(p_0_y[i],p_1_y[i],p_2_y[i],p_3_y[i]))

    overlap_gt_boxes[:,0] = x_min
    overlap_gt_boxes[:,1] = y_min
    overlap_gt_boxes[:,2] = x_max
    overlap_gt_boxes[:,3] = y_max

    return overlap_gt_boxes
