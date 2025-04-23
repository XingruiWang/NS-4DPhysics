#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import random
import cv2
import numpy as np
from tqdm import tqdm 
import json
from pycocotools.mask import encode
# import some common detectron2 utilities
from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from superclevr_detection import get_superclevr_dicts
from clevr_detection import get_clevr_dicts
# from detectron2.evaluation import (
#     CityscapesInstanceEvaluator,
#     CityscapesSemSegEvaluator,
#     COCOEvaluator,
#     COCOPanopticEvaluator,
#     DatasetEvaluators,
#     LVISEvaluator,
#     PascalVOCDetectionEvaluator,
#     SemSegEvaluator,
#     verify_results,
# )
# from detectron2.modeling import GeneralizedRCNNWithTTA


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    print(dataset_name)
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        print()
        evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
        # return build_evaluator(cfg, dataset_name, output_folder)
        return evaluator

    @classmethod
    def evaludate(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference for validation...")
        # model = GeneralizedRCNNWithTTA(cfg, model)

        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k: v for k, v in res.items()})

        return res

    @classmethod
    def test_superclevr(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running test...")
        # model = GeneralizedRCNNWithTTA(cfg, model)

        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "test")
            )
            for name in cfg.DATASETS.TEST
        ]

        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k: v for k, v in res.items()})

        return res

def config(args):
    cfg = get_cfg()
    
    EPOCH = 200
    TOTAL_NUM_IMAGES = 10500
    # TOTAL_NUM_IMAGES = 3200
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    if not args.multiple_dataset:
        cfg.DATASETS.TRAIN = ("{}_train".format(args.dataset_name),)
        cfg.DATASETS.VAL = ("{}_val".format(args.dataset_name),)
        cfg.DATASETS.TEST = ("{}_val".format(args.dataset_name),)
    else:
        cfg.DATASETS.TRAIN = ["{}_{}_train".format(args.dataset_name, i) for i in args.name_list]
        cfg.DATASETS.VAL = ["{}_{}_val".format(args.dataset_name, i) for i in args.name_list]
        cfg.DATASETS.TEST = ["{}_{}_val".format(args.dataset_name, i) for i in args.name_list]
    
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.DATALOADER.NUM_WORKERS = 2
    if args.load:
        cfg.MODEL.WEIGHTS = args.load
    else:
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.NUM_GPUS = 2

    single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
    # iterations_for_one_epoch = TOTAL_NUM_IMAGES / single_iteration
    cfg.SOLVER.ITER_EACH_EPOCH = TOTAL_NUM_IMAGES / single_iteration
    # cfg.TEST.EVAL_PERIOD = TOTAL_NUM_IMAGES / single_iteration
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.ITER_EACH_EPOCH * EPOCH)
    
    cfg.SOLVER.BASE_LR = 0.02  # pick a good LR
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    
    # cfg.SOLVER.MAX_ITER = 100000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = [20000, 30000, 50000]        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    
    if args.dataset_name == 'superclevr':
        if args.train_type == 'objects':
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27 
        elif args.train_type == 'objects_parts':
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 129
    elif args.dataset_name == 'clevr':
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 48
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    if args.dataset_name == 'superclevr':
        if args.multiple_dataset:
            cfg.OUTPUT_DIR = '../../data/{}/detection/{}'.format("multiple", args.train_type)
        else:
            cfg.OUTPUT_DIR = '../../data/{}/segmentation/{}'.format(args.set_name, args.train_type)
    elif args.dataset_name == 'clevr':
        cfg.OUTPUT_DIR = '../../data/{}/segmentation/{}'.format(args.dataset_name, args.train_type)

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

def register_dataset_multiple(name_list, type = 'objects', splits = ["train", "val"], name='superclevr'):
    if name == 'superclevr':
        get_dict, classes = get_superclevr_dicts(type)
        for set_name in name_list:
            for split in splits:
                print("Registering dataset of {} / {} / {} / {}".format(name, set_name, type, split))
                img_dir = "/mnt/data0/xingrui/ccvl17/{}/images".format(set_name)
                scene_file = "/mnt/data0/xingrui/ccvl17/{}/superCLEVR_scenes.json".format(set_name)
                
                DatasetCatalog.register("superclevr_{}_{}".format(set_name, split) , lambda split=split: get_dict(scene_file, img_dir, split, trim=0.3))
                MetadataCatalog.get("superclevr_{}_{}".format(set_name, split)).set(thing_classes=classes)

def register_dataset(name, scene_file, img_dir, type = 'objects', splits = ["train", "val"]):
    if name == 'superclevr':
        get_dict, classes = get_superclevr_dicts(type)
        for split in splits:
            print("Registering dataset of {} / {} / {}".format(name, type, split))
            DatasetCatalog.register("superclevr_" + split, lambda split=split: get_dict(scene_file, img_dir, split))
            MetadataCatalog.get("superclevr_" + split).set(thing_classes=classes)
    elif name == 'clevr':
        get_dict, classes = get_clevr_dicts()
        for split in splits:
            print("Registering dataset of {} / {} / {}".format(name, type, split))
            DatasetCatalog.register("clevr_" + split, lambda split=split: get_dict(scene_file, img_dir, split))
            MetadataCatalog.get("clevr_" + split).set(thing_classes=classes)        
def main(args):

    if args.eval_only:
        if args.multiple_dataset:
            register_dataset_multiple(name_list = args.name_list, 
            type = args.train_type,
            splits = ['val'])
        else:
            register_dataset(name = args.dataset_name, 
            scene_file = args.scene_file, 
            img_dir = args.img_dir,
            type = args.train_type,
            splits = ['val'])
    else:
        if args.multiple_dataset:
            register_dataset_multiple(name_list = args.name_list, 
            type = args.train_type,
            splits = ['train', 'val'])
        else:
            register_dataset(name = args.dataset_name, 
            scene_file = args.scene_file, 
            img_dir = args.img_dir,
            type = args.train_type)  
                    
    cfg = config(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test_superclevr(cfg, model)

        if comm.is_main_process():
            verify_results(cfg, res)

        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    print("Resume = ", args.resume)
    trainer.resume_or_load(resume=args.resume)
    # if cfg.TEST.AUG.ENABLED:
    trainer.register_hooks(
        [hooks.EvalHook(cfg.SOLVER.ITER_EACH_EPOCH, lambda: trainer.evaludate(cfg, trainer.model)),
        hooks.BestCheckpointer(cfg.SOLVER.ITER_EACH_EPOCH, trainer.checkpointer, val_metric = "bbox/AP50")]
    )
    
    return trainer.train()

def pred(args):
    print("Start Prediction")
    # register_dataset(name = "superclevr", 
    # scene_file = args.scene_file
    
    question_file = args.question_file
    img_dir = args.img_dir
    split = args.split
    start_img = 0
    end_img = int(os.listdir(args.img_dir)[-1].split(".")[0].split("_")[-1])
    end_img=100
    if question_file and not os.path.exists(question_file):
        print("Do not find path", question_file)
        print("Load question from 'ver_mask/questions/superCLEVR_questions_merged.json'")
        question_file = '/mnt/data0/xingrui/ccvl17/ver_mask/questions/superCLEVR_questions_merged.json'

    if question_file and split == 'test' and args.test_length > 0:
        print("Test length =", args.test_length)
        with open(question_file) as q_f:
            all_questions = json.load(q_f)['questions']
            start_img = all_questions[-args.test_length]['image_index']
            end_img = all_questions[-1]['image_index']
            print("Loading image of range:",start_img, end_img)

    _, classes = get_superclevr_dicts(args.train_type)

    metadata = MetadataCatalog.get("superclevr_{}".format(split)).set(thing_classes=classes)
    cfg = config(args)
    predictor = DefaultPredictor(cfg)

    output_result = {}
    scene_dict = {}
    for img_id in tqdm(range(start_img, end_img+1)):
        img_name = "superCLEVR_new_{}.png".format(str(img_id).zfill(6)) # 032809
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue
        scene_dict = {'file_name': img_path,
                    'objects': [],
                    }
        im = cv2.imread(img_path)
        outputs = predictor(im)
        pred_objects = outputs['instances'].get_fields()

        bboxes = pred_objects['pred_boxes'].tensor.tolist()
        scores = pred_objects['scores'].tolist()
        classes = pred_objects['pred_classes'].tolist()
        masks = pred_objects['pred_masks'].cpu().numpy().astype(np.uint8)

        for i in range(len(classes)):
            score = scores[i]
            if score < 0.7:
                continue
            box = bboxes[i]
            mask=encode(np.asfortranarray(masks[i]))
            mask['counts'] = str( mask['counts'])
            class_id = classes[i]
            scene_dict['objects'].append({'bbox': box,
                                        'class': class_id,
                                        'mask':mask,
                                        'score': score})
        output_result[img_id] = scene_dict

    if not args.split == 'test':
        with open(os.path.join(cfg.OUTPUT_DIR,'superclevr_objects_seg_pred.json'), 'w') as f:
            json.dump(output_result, f)
        print("Write to", os.path.join(cfg.OUTPUT_DIR,'superclevr_objects_seg_pred.json'))

    else:
        with open(os.path.join(cfg.OUTPUT_DIR,'superclevr_objects_seg_test.json'), 'w') as f:
            json.dump(output_result, f)
        print("Write to", os.path.join(cfg.OUTPUT_DIR,'superclevr_objects_seg_test.json'))

def pred_clevr(args):
    print("Start Prediction")
    # register_dataset(name = "superclevr", 
    # scene_file = args.scene_file
    start_img = 0
    end_img = max([int(n.split(".")[0].split("_")[-1]) for n in os.listdir(args.img_dir)])
   
    print(start_img, end_img)
    question_file = args.question_file
    img_dir = args.img_dir
    split = args.split

    if question_file and split == 'test':
        
        with open(question_file) as q_f:
            all_questions = json.load(q_f)['questions']
            print("Test length =", len(all_questions))
            start_img = all_questions[0]['image_index']
            end_img = all_questions[-1]['image_index']
            print("Loading image of range:",start_img, end_img)

    _, classes = get_clevr_dicts()

    metadata = MetadataCatalog.get("clevr_{}".format(split)).set(thing_classes=classes)
    cfg = config(args)
    predictor = DefaultPredictor(cfg)

    output_result = {}
    scene_dict = {}
    print(start_img, end_img)
    for img_id in tqdm(range(start_img, end_img)):
        img_name = "CLEVR_val_{}.png".format(str(img_id).zfill(6)) # 032809
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            break
        scene_dict = {'file_name': img_path,
                    'objects': [],
                    }
        im = cv2.imread(img_path)
        outputs = predictor(im)
        pred_objects = outputs['instances'].get_fields()

        bboxes = pred_objects['pred_boxes'].tensor.tolist()
        scores = pred_objects['scores'].tolist()
        classes = pred_objects['pred_classes'].tolist()

        masks = pred_objects['pred_masks'].cpu().numpy().astype(np.uint8)
        
        for i in range(len(classes)):
            score = scores[i]

            box = bboxes[i]
            class_id = classes[i]
    
            mask=encode(np.asfortranarray(masks[i]))
            mask['counts'] = str( mask['counts'])
            scene_dict['objects'].append({'bbox': box,
                                        'mask':mask,
                                        'class': class_id,
                                        'score': score})
        output_result[img_id] = scene_dict

    if "mini" in args.img_dir:
        with open(os.path.join(cfg.OUTPUT_DIR,'clevr-mini_objects_test-seg.json'), 'w') as f:
            json.dump(output_result, f)
        print("Write to", os.path.join(cfg.OUTPUT_DIR,'clevr-mini_objects_test-seg.json'))
    elif "CLEVR_v1.0" in args.img_dir:
        with open(os.path.join(cfg.OUTPUT_DIR,'clevr-val_objects_pred-seg.json'), 'w') as f:
            json.dump(output_result, f)
        print("Write to", os.path.join(cfg.OUTPUT_DIR,'clevr-val_objects_pred-seg.json'))


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--dataset_name', default='superclevr',  type=str, help='dataset_name')
    parser.add_argument('--train_type', default='objects', choices=['objects', 'parts', 'objects_parts'], type=str, help='experiment directory')
    parser.add_argument('--multiple_dataset', action='store_true')

    parser.add_argument('--export_objects', type=str, help='experiment directory')
    parser.add_argument('--scene_file', type=str, help='')
    parser.add_argument('--question_file', type=str, default='', help='')
    parser.add_argument('--img_dir', type=str, help='')
    parser.add_argument('--set_name', type=str, default='ver_mask', help='')
    parser.add_argument('--name_list', nargs='+')
    parser.add_argument('--load', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--test_length', default = -1, type=int)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--pred_bbox', action='store_true')
    parser.add_argument('--batch_size', default=8, type=int)
    
    args = parser.parse_args()

    print("Command Line Args:", args)
    if not args.pred_bbox:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
    else:
        if args.dataset_name == 'superclevr':
            pred(args)
        elif args.dataset_name == 'clevr':
            pred_clevr(args)