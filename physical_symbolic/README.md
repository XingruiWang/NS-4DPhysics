


# Probalistic NS-VQA

## Train model

### Step 1: Train MaskRCNN

- set_name. Datset name. E.g. `ver_mask`
- scene_file. The json file store the scene annotation of superclevr. 
    - E.g. /path_to_superclevr/ver_mask/superCLEVR_scenes.json
- img_dir. The folder of all superclevr images
    - E.g. /path_to_superclevr/ver_mask/images
- if train on clevr, use the similar path of clevr-mini, and change `dataset_name` to clevr

Run:

```shell

cd scene_parse/detection

python train_step.py \
    --num-gpus 1 \
    --set_name ver_mask \
    --train_type objects \
    --scene_file /path_to_superclevr/ver_mask/superCLEVR_scenes.json \
    --img_dir /path_to_superclevr/ver_mask/images \
    --dataset_name superclevr \
    --batch_size 4
```

### Step 2: Train Attributes Network

#### 1. If using groundtruth segmentation to extract objects

- `scene_path`. Same as `scene_file` before.
- `img_dir`. Same as before.
- `dataset`. Same as `dataset_name` before.
- 'obj_ann_path'. Output an annotated json file in the first time.

```
cd scene_parse/attr_net

python tools/run_train.py \
    --run_dir ../../data/ver_mask/attr_net/outputs/trained_model/objects \
    --obj_ann_path ../../data/ver_mask/attr_net/outputs/superclevr_aligned_gt.json \
    --img_dir /path_to_superclevr/ver_mask/images \
    --scene_path /path_to_superclevr/ver_mask/superCLEVR_scenes.json \
    --dataset superclevr \
    --type objects \
    --display_every 1 \
```

#### 2. If using the prediction mask from MaskRCNN to extract objects
 
(a) Generate a prediction of segmentation and align the groundtruth attributes annotation

```
cd scene_parse/detection
python train_step.py \
    --num-gpus 1 \
    --set_name ver_mask \
    --train_type objects \
    --split all \
    --scene_file /path_to_superclevr/ver_mask/superCLEVR_scenes.json \
    --img_dir /path_to_superclevr/ver_mask/images \
    --load ../../data/ver_mask/segmentation/objects/model_best.pth \
    --pred_bbox
    
python align_objects.py \
    --pred_bbox_file ../../data/ver_mask/segmentation/objects/superclevr_objects_seg_pred.json \
    --scene_anno_file ../../data/ver_mask/attr_net/outputs/superclevr_aligned_gt.json \
    --output_file ../../data/ver_mask/attr_net/outputs/superclevr_aligned.json
```

(b) Train attributes network

- change the `obj_ann_path` to the `output_file` of (a)

```
cd scene_parse/attr_net

python tools/run_train.py \
    --run_dir ../../data/ver_mask/attr_net/outputs/trained_model/objects \
    --obj_ann_path ../../data/ver_mask/attr_net/outputs/superclevr_aligned.json \
    --img_dir /path_to_superclevr/ver_mask/images \
    --scene_path /path_to_superclevr/ver_mask/superCLEVR_scenes.json \
    --dataset superclevr \
    --type objects \
    --display_every 1 \
```

## Prediction

The train and test domain can be different.

```
DATASET=ver_mask # set name (domain) for test
TRAIN_SET=ver_mask # set name (domain) where trained the model
```
### Step 1: Object detection / segmentation

- Use `pred_bbox` to predict
- `scene_file`, `img_dir`, `question_file`. The scene json, image folders, and question json file of SuperCLEVR
- `load`. Th path of trained MaskRCNN checkpoint. Can be changed if testing OOD cases

```
cd scene_parse/detection/

CUDA_VISIBLE_DEVICES=3 python train_step.py \
    --num-gpus 1 \
    --set_name ver_mask \
    --train_type objects \
    --scene_file  /path_to_superclevr/${DATASET}/superCLEVR_scenes.json \
    --question_file  /path_to_superclevr/${DATASET}/superCLEVR_questions_merged.json \
    --img_dir /path_to_superclevr/${DATASET}/images \
    --load ../../data/${TRAIN_DATASET}/segmentation/objects/model_best.pth \
    --pred_bbox \
    --split test \
    --test_length 10000
```


### Step 2. Run attributes prediction

(a) Using probability (P-NSVQA)

`pred_bbox`. Generation of the prdiction of the bbox and segmentation from step 1.

```
cd scene_parse/attr_net

python tools/run_test_prob.py \
    --load_path ../../data/${TRAIN_DATASET}/attr_net/outputs/trained_model/objects/checkpoint_best.pt \
    --img_dir /path_to_superclevr/${DATASET}/images \
    --pred_bbox ../../data/${DATASET}/segmentation/objects/superclevr_objects_test.json \
    --output_file ../../data/${DATASET}/reason/scene_pred_${TRAIN_DATASET}_${DATASET}_prob.json
```



(b) Using original NSVQA

```
cd scene_parse/attr_net

python tools/run_test.py \
    --load_path ../../data/${TRAIN_DATASET}/attr_net/outputs/trained_model/objects/checkpoint_best.pt \
    --img_dir /path_to_superclevr/${DATASET}/images \
    --pred_bbox ../../data/${DATASET}/segmentation/objects/superclevr_objects_test.json \
    --output_file ../../data/${DATASET}/reason/scene_pred_${TRAIN_DATASET}_${DATASET}.json
```

### Step 3. Run reasoning

(a) Preprocess Question

```
mkdir -p ../data/${DATASET}/preprocess
python tools/preprocess_questions_superclevr.py \
    --input_questions_json /path_to_superclevr/${DATASET}/superCLEVR_questions_merged.json \
    --output_h5_file ../data/${DATASET}/preprocess/SuperCLEVR_questions.h5 \
    --output_vocab_json ../data/${DATASET}/preprocess/SuperCLEVR_vocab.json
```

(b).i Run P-NSVQA

```
cd $reason
mkdir -p ../data/${DATASET}/reason

python tools/run_test_prob.py \
    --superclevr_question_path ../data/${DATASET}/preprocess/SuperCLEVR_questions.h5 \
    --superclevr_scene_path ../data/${DATASET}/reason/scene_pred_${TRAIN_DATASET}_${DATASET}_prob.json \
    --superclevr_vocab_path ../data/${DATASET}/preprocess/SuperCLEVR_vocab.json \
    --superclevr_gt_question_path /path_to_superclevr/${DATASET}/superCLEVR_questions_merged.json \
    --save_result_path ../data/${DATASET}/reason/results-train_on_${TRAIN_DATASET}-test_on_${DATASET}.json \
    --length 10000 \
    --prob \
````

(b).2 Run NSVQA

```
cd $reason
mkdir -p ../data/${DATASET}/reason

python tools/run_test.py \
    --superclevr_question_path ../data/${DATASET}/preprocess/SuperCLEVR_questions.h5 \
    --superclevr_scene_path ../data/${DATASET}/reason/scene_pred_${TRAIN_DATASET}_${DATASET}.json \
    --superclevr_vocab_path ../data/${DATASET}/preprocess/SuperCLEVR_vocab.json \
    --superclevr_gt_question_path /path_to_superclevr/${DATASET}/superCLEVR_questions_merged.json \
    --save_result_path ../data/${DATASET}/reason/results-train_on_${TRAIN_DATASET}-test_on_${DATASET}.json \
    --length 10000 \
````
