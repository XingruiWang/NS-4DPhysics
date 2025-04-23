import json
import os
import sys
import pycocotools.mask as mask_util
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np


import random
import pdb
from tqdm import tqdm


SUPERCLEVR_SHAPES = ['car', 'suv', 'wagon', 'minivan', 'sedan', 'truck', 'addi', 'bus', 'articulated', 'regular', 'double', 'school', 'motorbike', 'chopper', 'dirtbike', 'scooter', 'cruiser', 'aeroplane', 'jet', 'fighter', 'biplane', 'airliner', 'bicycle', 'road', 'utility', 'mountain', 'tandem']
SUPERCLEVR_PARTNAMES = ['left_mirror', 'fender_front', 'footrest', 'wheel_front_right', 'crank_arm_left', 'wheel_front_left', 'bumper', 'headlight', 'door_front_left', 'wing', 'front_left_wheel', 'side_stand', 'footrest_left_s', 'tailplane_left', 'wheel_front', 'mirror', 'right_head_light', 'back_left_door', 'left_tail_light', 'head_light_right', 'gas_tank', 'front_bumper', 'tailplane', 'taillight_center', 'back_bumper', 'headlight_right', 'panel', 'front_right_door', 'door_mid_left', 'hood', 'door_left_s', 'front_right_wheel', 'wing_left', 'head_light_left', 'back_right_door', 'tail_light_right', 'seat', 'taillight', 'door_front_right', 'trunk', 'back_left_wheel', 'exhaust_right_s', 'cover', 'brake_system', 'wing_right', 'pedal_left', 'rearlight', 'headlight_left', 'right_tail_light', 'engine_left', 'crank_arm', 'fender_back', 'engine', 'fender', 'door_back_right', 'wheel_back_left', 'back_license_plate', 'cover_front', 'headlight_center', 'engine_right', 'roof', 'left_head_light', 'taillight_right', 'fin', 'saddle', 'mirror_right', 'door', 'bumper_front', 'door_mid_right', 'head_light', 'bumper_back', 'wheel_back_right', 'footrest_right_s', 'drive_chain', 'license_plate_back', 'tail_light', 'pedal', 'windscreen', 'license_plate', 'exhaust_left_s', 'handle_left', 'handle', 'back_right_wheel', 'right_mirror', 'wheel', 'fork', 'taillight_left', 'handle_right', 'front_left_door', 'carrier', 'license_plate_front', 'crank_arm_right', 'wheel_back', 'cover_back', 'propeller', 'exhaust', 'tail_light_left', 'mirror_left', 'pedal_right', 'tailplane_right', 'door_right_s', 'front_license_plate']



def str_to_biimg(imgstr):
    img=[]
    cur = 0

    for num in imgstr.strip().split(','):
        num = int(num)
        img += [cur] * num
        cur = 1 - cur
    img = np.array(img).astype(np.uint8)
    img = np.asfortranarray(img.reshape((480, 640)))

    # cv2.imwrite("output/test_mask.png", img*255)
    return img
    
# /mnt/data0/xingrui/superclevr_anno/superclevr_anno.json
def get_superclevr_object_dicts(scene_file, img_dir, split_set = 'train', range_idx = None, trim=1.0):

    class_counter = {}
    split = [0.5, 0.1, 0.4]
    shape2id = {n: i for i, n in enumerate(SUPERCLEVR_SHAPES)}
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
            record["height"] = 480
            record["width"] = 640
        
            # annos = v["regions"]
            objs = []
            for i, objects in enumerate(v['objects']):
                object_mask_box = v['obj_mask_box'][str(i)]['obj']
                # mask = [int(i) for i in object_mask_box[1].split(',')][1:]
                mask = str_to_biimg(object_mask_box[1])
                mask = mask_util.encode(mask)
                # print(mask)
                obj = {
                    "bbox": object_mask_box[0],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": mask,
                    "category_id": shape2id[objects['shape']],
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
            record["height"] = 480
            record["width"] = 640
        
            # annos = v["regions"]
            objs = []
            for i, objects in enumerate(v['objects']):
                object_mask_box = v['obj_mask_box'][str(i)]['obj']
                bbox = object_mask_box[0]

                class_counter[objects['shape']] = class_counter.get(objects['shape'], 0) + 1
                mask = str_to_biimg(object_mask_box[1])
                mask = mask_util.encode(mask)
                
                obj = {
                    "bbox": object_mask_box[0],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": mask,
                    "category_id": shape2id[objects['shape']],
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
            record["file_name"] = os.path.join(img_dir, v["image_filename"])
            record["image_id"] = v["image_index"]
            record["height"] = 480
            record["width"] = 640
        
            # annos = v["regions"]
            objs = []
            for i, objects in enumerate(v['objects']):
                object_mask_box = v['obj_mask_box'][str(i)]['obj']
                bbox = object_mask_box[0]
                # print()

                mask = str_to_biimg(object_mask_box[1])
                mask = mask_util.encode(mask)
                # print(object_mask_box[1])
                
                obj = {
                    "bbox": object_mask_box[0],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": mask,
                    "category_id": shape2id[objects['shape']],
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
                object_mask_box = v['obj_mask_box'][str(i)]['obj']
                bbox = object_mask_box[0]
                mask = str_to_biimg(object_mask_box[1])
                mask = mask_util.encode(mask)
                obj = {
                    "bbox": object_mask_box[0],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": mask,
                    "category_id": shape2id[objects['shape']],
                }
                objs.append(obj)
            record["annotations"] = objs
            test_dicts.append(record)
        return test_dicts        
    else:
        raise ValueError("Split invalid")


def get_superclevr_parts_dicts(scene_file, img_dir, split_set = 'train', range_idx = None):

    class_counter = {}
    split = [0.5, 0.1, 0.4]
    shape2id = {n: i for i, n in enumerate(SUPERCLEVR_PARTNAMES)}
    with open(scene_file) as f:
        imgs_anns = json.load(f)
    
    L = len(imgs_anns['scenes'])
    print("TRAIN SET LENGTH:", int(split[0] * L))
    print("VALIDATION SET LENGTH:", int(split[1] * L))

    if split_set == 'train':
        set_length = int(split[0] * L)
        start_id = 0
    elif split_set == 'val':
        set_length = int(split[1] * L)
        start_id = int(split[0] * L)      
    elif split_set == 'test':
        set_length = int(split[2] * L)
        start_id = int(split[0] * L) + int(split[1] * L)       
    else:
        raise ValueError("Split invalid")

    end_id = start_id + set_length

    data_dicts = []
    for scene_id, v in enumerate(imgs_anns['scenes'][start_id:end_id]):
    # for scene_id, v in enumerate(imgs_anns['scenes'][:100]):
        record = {}
        record["file_name"] = os.path.join(img_dir, v["image_filename"])
        record["image_id"] = v["image_index"]
        record["height"] = 480
        record["width"] = 640
    
        # annos = v["regions"]
        parts = []
        for i, objects in enumerate(v['objects']):
            for p_name in objects['parts']:


                if p_name not in  v['obj_mask_box'][str(i)]:
                    continue

                part_mask_box = v['obj_mask_box'][str(i)][p_name]
                # mask = [int(i) for i in object_mask_box[1].split(',')][1:]

                if sum(part_mask_box[0]) == 0:
                    continue            

                part = {
                    "bbox": part_mask_box[0],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    # "segmentation": [mask],
                    "category_id": shape2id[p_name],
                }
                parts.append(part)
        record["annotations"] = parts
        data_dicts.append(record)
    return data_dicts

def get_superclevr_objects_parts_dicts(scene_file, img_dir, split_set = 'train'):

    class_counter = {}
    split = [0.5, 0.1, 0.4]
    shape2id = {n: i for i, n in enumerate(SUPERCLEVR_SHAPES+SUPERCLEVR_PARTNAMES)}
    with open(scene_file) as f:
        imgs_anns = json.load(f)
    
    L = len(imgs_anns['scenes'])
    print("TRAIN SET LENGTH:", int(split[0] * L))
    print("VALIDATION SET LENGTH:", int(split[1] * L))

    if split_set == 'train':
        set_length = int(split[0] * L)
        start_id = 0
    elif split_set == 'val':
        set_length = int(split[1] * L)
        start_id = int(split[0] * L)      
    elif split_set == 'test':
        set_length = int(split[2] * L)
        start_id = int(split[0] * L) + int(split[1] * L)       
    else:
        raise ValueError("Split invalid")

    end_id = start_id + set_length

    data_dicts = []
    for scene_id, v in enumerate(imgs_anns['scenes'][start_id:end_id]):
    # for scene_id, v in enumerate(imgs_anns['scenes'][:100]):
        record = {}
        record["file_name"] = os.path.join(img_dir, v["image_filename"])
        record["image_id"] = v["image_index"]
        record["height"] = 480
        record["width"] = 640
    
        # annos = v["regions"]
        objects_parts = []
        for i, objects in enumerate(v['objects']):
            # objects
            object_mask_box = v['obj_mask_box'][str(i)]['obj']
            bbox = object_mask_box[0]
            # mask = object_mask_box[1]
            obj = {
                "bbox": object_mask_box[0],
                "bbox_mode": BoxMode.XYWH_ABS,
                # "segmentation": object_mask_box[1],
                "category_id": shape2id[objects['shape']],
            }
            objects_parts.append(obj)
            # parts
            for p_name in objects['parts']:


                if p_name not in  v['obj_mask_box'][str(i)]:
                    continue

                part_mask_box = v['obj_mask_box'][str(i)][p_name]
                # mask = [int(i) for i in object_mask_box[1].split(',')][1:]

                if sum(part_mask_box[0]) == 0:
                    continue            

                part = {
                    "bbox": part_mask_box[0],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    # "segmentation": [mask],
                    "category_id": shape2id[p_name],
                }
                objects_parts.append(part)
        record["annotations"] = objects_parts
        data_dicts.append(record)
    return data_dicts


def get_superclevr_dicts(type):

    if type == 'objects':
        return get_superclevr_object_dicts, SUPERCLEVR_SHAPES
    elif type == 'parts':
        return get_superclevr_parts_dicts, SUPERCLEVR_PARTNAMES
    elif type == 'objects_parts':
        return get_superclevr_objects_parts_dicts, SUPERCLEVR_SHAPES + SUPERCLEVR_PARTNAMES
    else:
        raise ValueError("Type = {} is invalid".format(type))
        

if __name__ == '__main__':
    import cv2
    from detectron2.utils.visualizer import Visualizer

    scene_file = '/mnt/data0/xingrui/ccvl17/ver_mask/superCLEVR_scenes.json'
    img_dir = '/mnt/data0/xingrui/ccvl17/ver_mask/images'

    # dataset_dicts = get_superclevr_parts_dicts(scene_file, img_dir, split_set='test')
    dataset_dicts = get_superclevr_object_dicts(scene_file, img_dir, split_set='train', trim=0.1)

    i = 0
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1])
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite("output/test_{}.png".format(i), out.get_image()[:, :, ::-1])  
        i+=1  
    