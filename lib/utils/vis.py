# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import cv2
import numpy as np
import torchvision

from dataset import VIS_CONFIG


def add_joints(image, joints, color, dataset='COCO'):
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    2
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image


def save_valid_image(image, joints, file_name, dataset='COCO'):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for person in joints:
        color = np.random.randint(0, 255, size=3)
        color = [int(i) for i in color]
        add_joints(image, person, color, dataset=dataset)

    cv2.imwrite(file_name, image)


def make_heatmaps(image, heatmaps):
    heatmaps = heatmaps.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = tagmap.add(-min)\
                       .div(max - min + 1e-5)\
                       .mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .cpu()\
                       .numpy()

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap*0.9 + image_resized*0.1

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(
                        ndarr,
                        (int(joint[0]), int(joint[1])),
                        2,
                        [255, 0, 0],
                        2
                    )
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_maps(
        batch_image,
        batch_maps,
        batch_mask,
        file_name,
        map_type='heatmap',
        normalize=True
):
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_maps.size(0)
    num_joints = batch_maps.size(1)
    map_height = batch_maps.size(2)
    map_width = batch_maps.size(3)

    grid_image = np.zeros(
        (batch_size*map_height, (num_joints+1)*map_width, 3),
        dtype=np.uint8
    )

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        maps = batch_maps[i]

        if map_type == 'heatmap':
            image_with_hms = make_heatmaps(image, maps)
        elif map_type == 'tagmap':
            image_with_hms = make_tagmaps(image, maps)

        height_begin = map_height * i
        height_end = map_height * (i + 1)

        grid_image[height_begin:height_end, :, :] = image_with_hms
        if batch_mask is not None:
            mask = np.expand_dims(batch_mask[i].byte().cpu().numpy(), -1)
            grid_image[height_begin:height_end, :map_width, :] = \
                grid_image[height_begin:height_end, :map_width, :] * mask

    cv2.imwrite(file_name, grid_image)


def save_debug_images(
    config,
    batch_images,
    batch_heatmaps,
    batch_masks,
    batch_outputs,
    prefix
):
    if not config.DEBUG.DEBUG:
        return

    num_joints = config.DATASET.NUM_JOINTS
    batch_pred_heatmaps = batch_outputs[:, :num_joints, :, :]
    batch_pred_tagmaps = batch_outputs[:, num_joints:, :, :]

    if config.DEBUG.SAVE_HEATMAPS_GT and batch_heatmaps is not None:
        file_name = '{}_hm_gt.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_heatmaps, batch_masks, file_name, 'heatmap'
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        file_name = '{}_hm_pred.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_pred_heatmaps, batch_masks, file_name, 'heatmap'
        )
    if config.DEBUG.SAVE_TAGMAPS_PRED:
        file_name = '{}_tag_pred.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_pred_tagmaps, batch_masks, file_name, 'tagmap'
        )

skeleton_colors = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
    [255, 85, 85],
    [255, 170, 85],
    [255, 170, 170],
]

coco_skeleton_pairs = [
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [6, 8], [7, 9], [8,10],
    [5, 11],[6, 12], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]
]

def draw_skeleton(canvas, kpts, stickwidth=3, skeleton_pairs=coco_skeleton_pairs):
    num_persons = kpts.shape[0]
    num_skeletons = len(skeleton_pairs)

    for i in range(num_skeletons):
        for n in range(num_persons):
            cur_canvas = canvas.copy()
            index = np.array(skeleton_pairs[i])

            vis = kpts[n, index.astype(int), -1]
            if vis[0] == 0 or vis[1] == 0:
                continue

            Y = kpts[n, index.astype(int), 0]
            X = kpts[n, index.astype(int), 1]

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, skeleton_colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas