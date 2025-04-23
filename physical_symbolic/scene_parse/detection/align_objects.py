import 
import numpy
import json
from tqdm import tqdm
import random
from pycocotools.mask import decode
import pycocotools.mask as mask_util
import numpy as np
import copy
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--pred_bbox_file', type=str)
parser.add_argument('--scene_anno_file', type=str)
parser.add_argument('--output_file', type=str)
args = parser.parse_args()


DATASET = 'superclevr'
pred_bbox_file = args.pred_bbox_file
scene_anno_file = args.scene_anno_file
output_file = args.output_file

def _preprocess_rle(rle):
    """Turn ASCII string into rle bytes"""
    rle['counts'] = rle['counts'].encode('ASCII')
    return rle

def _to_ascii(rle):
    """Turn ASCII string into rle bytes"""
    rle['counts'] = eval(rle['counts'])
    return rle

def get_mask(rle):
    """Turn ASCII string into rle bytes"""
    mask = decode(rle)

def convert_bbox(bbox, format):
    if format == 'XYWH':
        x1, y1, w, h = bbox
        return {'x1': x1, 'x2': x1 + w, 'y1': y1, 'y2': y1 + h}
    elif format == 'XYXY':
        x1, y1, x2, y2 = bbox
        return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}        
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = convert_bbox(bb1, 'XYXY')
    bb2 = convert_bbox(bb2, 'XYWH')
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou_seg(seg1, seg2):

    intersection = seg1 * seg2
    union = 1 - (1 -seg1) * (1 - seg2)

    return np.sum(intersection) / np.sum(union)
 

with open(pred_bbox_file) as f:
    pred_bbox = json.load(f)

with open(scene_anno_file) as f:
    scene_anno = json.load(f)
# print(scene_anno.keys())

scene_anno_pred = {}


# for image_id in tqdm(random.sample(scene_anno.keys(), 1)):
for image_id in tqdm(pred_bbox):
    scene_anno_pred[image_id] = []

    pred_scene = pred_bbox[image_id]['objects']

    for o in pred_scene:
        bbox = o['bbox']
        score = o['score']
        class_id = o['class']
        seg_encode = copy.deepcopy(o['mask'])

        mask = np.array(mask_util.decode(_to_ascii(seg_encode)), dtype=np.float32)

        max_iou = 0.5
        max_id = -1
        for i, gt_object in enumerate(scene_anno[image_id]):

            # mask_gt = _preprocess_rle(gt_object['mask'])
            mask_gt = np.array(mask_util.decode(gt_object['mask']), dtype=np.float32)

            iou = get_iou_seg(mask, mask_gt)
            if iou > max_iou:
                max_id = i
                max_iou = iou

        if max_id > 0:

            aligned_object = scene_anno[image_id][max_id]
            scene_anno_pred[image_id].append({'image_filename': aligned_object['image_filename'],
                                                'shape': aligned_object['shape'],
                                                'color': aligned_object['color'],
                                                'material': aligned_object['material'],
                                                'size': aligned_object['size'],
                                                'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                                                'pred_class': class_id,
                                                'score': score,
                                                'mask': o['mask'] 
                                            })
    # print('pred', image_id, scene_anno_pred[image_id])

with open(output_file, 'w') as f:
    json.dump(scene_anno_pred, f)
    print("Writing to file:", output_file)
    
        # print(o['objects'])
#     image_filename = scene['image_filename']
#     objects = scene['objects']
#     obj_mask_box = scene['obj_mask_box']

#     pred_bbox_i = pred_bbox[image_id]

