# Visual complexity
# ver_nopart: ver_mask without part (easy)
# ver_mask: default version (mid, rd, co-1)
# ver_texture: ver_mask with texture (hard, bal)

### ver_texture_same: ver_texture, but with same object placement as in ver_mask (not used for now)

# Question redundancy
# ver_mask/no_redundant: reduce question redundancy in ver_mask (rd-)
# ver_mask
# ver_mask/rdn1: add question redundancy in ver_mask (rd+)

# Concept distribution.
# dist_texture: long tailed concept distribution, based on ver_texture (long)
# dist_texture_a1: slightly unbalanced concept distribution, based on ver_texture (slt)
# dist_texture_test: for testing the concept distributions (head, tail, oppo)
# codist: different concept distribtuion, based on ver_mask (co-0)
# codist_super: different concept distribtuion, based on ver_mask (co-2)

DATASET=ver_mask
# DATASET=ver_texture # -- detection --attribute
# DATASET=ver_nopart #50177 detection done, attr running
# DATASET=dist_texture # 50179,  done
# DATASET=dist_texture_a1 # 50183  done
# DATASET=dist_texture_test/head # 50187, detection done, attr no
# DATASET=dist_texture_test/tail # detection done, attr no
# DATASET=dist_texture_test/oppo # done
# DATASET=codist # detection no, attr running
# DATASET=codist_super # detection done, attr no

TYPE=objects

# 1. Detection
cd detection

## 1.1 Train detection
CUDA_VISIBLE_DEVICES=1 python train_step.py \
    --num-gpus 1 \
    --set_name ${DATASET} \
    --train_type ${TYPE} \
    --scene_file /mnt/data0/xingrui/ccvl17/${DATASET}/superCLEVR_scenes.json \
    --img_dir /mnt/data0/xingrui/ccvl17/${DATASET}/images \
    --dataset_name superclevr \
    --batch_size 4

## 2. Pred object using trained detection
CUDA_VISIBLE_DEVICES=1 python train_step.py \
    --num-gpus 1 \
    --set_name ${DATASET} \
    --train_type ${TYPE} \
    --split all \
    --scene_file /mnt/data0/xingrui/ccvl17/${DATASET}/superCLEVR_scenes.json \
    --img_dir /mnt/data0/xingrui/ccvl17/${DATASET}/images \
    --load /home/xingrui/vqa/ns-vqa/data/${DATASET}/segmentation/${TYPE}/model_best.pth \
    --pred_bbox

python align_objects.py \
    --pred_bbox_file /home/xingrui/vqa/ns-vqa/data/${DATASET}/segmentation/objects/superclevr_objects_seg_pred.json \
    --scene_anno_file /home/xingrui/vqa/ns-vqa/data/${DATASET}/attr_net/outputs/superclevr_aligned_gt.json \
    --output_file /home/xingrui/vqa/ns-vqa/data/${DATASET}/attr_net/outputs/superclevr_aligned.json

# 2. Attributes classification
cd attr_net
# If using gt bbox
CUDA_VISIBLE_DEVICES=0 python tools/run_train.py \
    --run_dir ../../data/${DATASET}/attr_net/outputs/trained_model/${TYPE} \
    --obj_ann_path ../../data/${DATASET}/attr_net/outputs/superclevr_aligned_gt.json \
    --img_dir /mnt/data0/xingrui/ccvl17/${DATASET}/images \
    --scene_path /mnt/data0/xingrui/ccvl17/${DATASET}/superCLEVR_scenes.json \
    --dataset superclevr \
    --type ${TYPE} \
    --display_every 1 \
    --resume ../../data/${DATASET}/attr_net/outputs/trained_model/${TYPE}/checkpoint_best.pt
cd ..
