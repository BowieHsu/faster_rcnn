# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import cv2

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

#array([ x,  y,  w, h, theta])

#def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
#                     scales=2**np.arange(3, 6)):
#    """
#    Generate anchor (reference) windows by enumerating aspect ratios X
#    scales wrt a reference (0, 0, 15, 15) window.
#    """

#    base_anchor = np.array([1, 1, base_size, base_size]) - 1
#    ratio_anchors = _ratio_enum(base_anchor, ratios)
#    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
#                         for i in xrange(ratio_anchors.shape[0])])

#    anchors_angle = func_angle(anchors)

#    return anchors

PIE_180 = np.pi / 180.0

def generate_anchors(base_size=16, ratios=[1,2],
                     scales=2**np.arange(3,6),
                     #angles = [-30, -15, 0, 15, 30, 45, 60, 75, 90, 105, 120]):
                     angles = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]):
                    #  angles = [-90, -80, -70, -80, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _non_ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    anchors = np.vstack([_angle_enum(anchors[i, :], angles)
                         for i in xrange(anchors.shape[0])])
    print(anchors.shape)
    print(anchors)
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _non_ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)

    # print(w, h, x_ctr, y_ctr)

    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    # print('ws', ws, 'hs', hs, 'anchors', anchors)
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)

    # print(w, h, x_ctr, y_ctr)

    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    # print('ws', ws, 'hs', hs, 'anchors', anchors)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _angle_enum(anchor, angles):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    angle_anchors = []
    for angle in angles:
        angle = round(angle * PIE_180, 3)
        #angle_anchors.append([anchor[0], anchor[1],anchor[2],anchor[3],angle])
        angle_anchors.append([x_ctr, y_ctr, w, h, angle])
    return angle_anchors



if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    # a = generate_anchors(base_size=32)
    print time.time() - t
    print a
    from IPython import embed; embed()
