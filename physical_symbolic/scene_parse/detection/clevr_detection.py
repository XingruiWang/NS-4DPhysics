import json
import os
import sys
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

import random
import pdb
from tqdm import tqdm
import pycocotools.mask as mask_util

import numpy as np



CLEVR_COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
CLEVR_MATERIALS = ['rubber', 'metal']
CLEVR_SHAPES = ['cube', 'cylinder', 'sphere']


def _preprocess_rle(rle):
    """Turn ASCII string into rle bytes"""
    rle['counts'] = rle['counts'].encode('ASCII')
    return rle

def seg2bbox(masks):
    """Computes the bounding box of each mask in a list of RLE encoded masks."""

    mask = np.array(mask_util.decode(masks), dtype=np.float32)

    def get_bounds(flat_mask):
        inds = np.where(flat_mask > 0)[0]
        return inds.min(), inds.max()


    flat_mask = mask.sum(axis=0)
    x0, x1 = get_bounds(flat_mask)
    flat_mask = mask.sum(axis=1)
    y0, y1 = get_bounds(flat_mask)
    # boxes = np.array([x0, y0, x1, y1])
    boxes = [x0, y0, x1, y1]

    return boxes

def combine_cls():
    cls_names = []
    for c in CLEVR_COLORS:
        for m in CLEVR_MATERIALS:
            for s in CLEVR_SHAPES:
                cls_names.append("{}-{}-{}".format(c, m, s))
    return cls_names

CLEVR_CLASSES = combine_cls()

# /mnt/data0/xingrui/superclevr_anno/superclevr_anno.json
def get_clevr_object_dicts(scene_file, img_dir, split_set = 'train', range_idx = None,  trim=1.0):

    class_counter = {}
    split = [0.8, 0.2, 0.0]
    class2id = {n: i for i, n in enumerate(CLEVR_CLASSES)}
    with open(scene_file) as f:
        imgs_anns = json.load(f)
    
    L = len(imgs_anns['scenes'])
    print("TRAIN SET LENGTH:", int(split[0] * L))
    print("VALIDATION SET LENGTH:", int(split[1] * L))

    if split_set == 'train':
        train_length = int(split[0] * L * trim)
        train_dicts = []
        for scene_id, v in enumerate(tqdm(imgs_anns['scenes'][:train_length])):
        # for scene_id, v in enumerate(imgs_anns['scenes'][:100]):
            record = {}
            record["file_name"] = os.path.join(img_dir, v["image_filename"])
            record["image_id"] = v["image_index"]
            record["height"] = 320
            record["width"] = 480
        
            # annos = v["regions"]
            objs = []
            for i, objects in enumerate(v['objects']):

                mask = _preprocess_rle(objects['mask'])
                bbox = seg2bbox(mask)
                
                # print(type(mask),mask)
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": mask,
                    "category_id": class2id["{}-{}-{}".format(objects['color'], objects['material'], objects['shape'])],
                }
                objs.append(obj)
            record["annotations"] = objs
            train_dicts.append(record)
        return train_dicts

    elif split_set == 'val':
        train_length = int(split[0] * L)
        val_length = int(split[1] * L * trim)

        val_dicts = []
        for scene_id, v in enumerate(tqdm(imgs_anns['scenes'][train_length: (train_length + val_length)])):
            record = {}
            record["file_name"] = os.path.join(img_dir, v["image_filename"])
            record["image_id"] = v["image_index"]
            record["height"] = 320
            record["width"] = 480
        
            # annos = v["regions"]
            objs = []
            for i, objects in enumerate(v['objects']):
                mask = _preprocess_rle(objects['mask'])
                bbox = seg2bbox(mask)

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": mask,
                    "category_id": class2id["{}-{}-{}".format(objects['color'], objects['material'], objects['shape'])],
                }
                objs.append(obj)
            record["annotations"] = objs
            val_dicts.append(record)            
        # if class_counter:
        #     print("Classes Counter:")
        #     for c,n in class_counter.items():
        #         print("class:", c, "| count:", n)
        
        return val_dicts

    elif split_set == 'test':
        if range_idx:
            test_scene = imgs_anns['scenes'][range_idx[0]: range_idx[1]]
        else:
            test_length = int(split[2] * L * trim)
            test_scene = imgs_anns['scenes'][-test_length:]
        print("Load test scene of length:", len(test_scene))
        test_dicts = []

        for scene_id, v in enumerate(test_scene):
            record = {}
            print(v["image_filename"])
            record["file_name"] = os.path.join(img_dir, v["image_filename"])
            record["image_id"] = v["image_index"]
            record["height"] = 320
            record["width"] = 480
        
            # annos = v["regions"]
            objs = []
            for i, objects in enumerate(v['objects']):
                mask = _preprocess_rle(objects['mask'])
                bbox = seg2bbox(mask)

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": mask,
                    "category_id": class2id["{}-{}-{}".format(objects['color'], objects['material'], objects['shape'])],
                }
                objs.append(obj)
            record["annotations"] = objs
            test_dicts.append(record)
        return test_dicts

    elif split_set == 'all':

        test_dicts = []

        for scene_id, v in enumerate(imgs_anns['scenes']):
            record = {}
            record["file_name"] = os.path.join(img_dir, v["image_filename"])
            record["image_id"] = v["image_index"]
            record["height"] = 480
            record["width"] = 640
        
            # annos = v["regions"]
            objs = []
            for i, objects in enumerate(v['objects']):
                mask = _preprocess_rle(objects['mask'])
                bbox = seg2bbox(mask)
                print(bbox)

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": object_mask_box[1],
                    "category_id": class2id["{}-{}-{}".format(objects['color'], objects['material'], objects['shape'])],
                }
                objs.append(obj)
            record["annotations"] = objs
            test_dicts.append(record)
        return test_dicts        
    else:
        raise ValueError("Split invalid")

def get_clevr_dicts():
    return get_clevr_object_dicts, CLEVR_CLASSES
        

if __name__ == '__main__':
    import cv2
    from detectron2.utils.visualizer import Visualizer

    scene_file = '/home/xingrui/data/CLEVR_mini/CLEVR_mini_coco_anns.json'
    img_dir = '/home/xingrui/data/CLEVR_mini/images'
    dataset_dicts = get_clevr_object_dicts(scene_file, img_dir, split_set='test')
    i = 0
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1])
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite("output/test_{}.png".format(i), out.get_image()[:, :, ::-1])  
        i+=1  
    