BASE=/home/xingrui/vqa/ns-vqa
TYPE=objects

# 1. Detection

## 1.1 Train detection
cd ${BASE}/scene_parse/detection/
CUDA_VISIBLE_DEVICES=1 python train_step.py \
    --num-gpus 1 \
    --dataset_name clevr \
    --train_type ${TYPE} \
    --scene_file /home/xingrui/data/CLEVR_mini/CLEVR_mini_coco_anns.json \
    --img_dir /home/xingrui/data/CLEVR_mini/images \
    --batch_size 4 \

### 2. Pred object using trained detection

CUDA_VISIBLE_DEVICES=0 python train_step.py \
    --num-gpus 1 \
    --dataset_name clevr \
    --set_name clevr \
    --train_type ${TYPE} \
    --question_file /home/xingrui/data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
    --img_dir /home/xingrui/data/CLEVR_mini/images \
    --scene_file /home/xingrui/data/CLEVR_mini/CLEVR_mini_coco_anns.json  \
    --load /home/xingrui/vqa/ns-vqa/data/clevr/segmentation/${TYPE}/model_best.pth \
    --pred_bbox \
    --split train

# 2. Attributes classification
cd attr_net
# If using gt bbox
CUDA_VISIBLE_DEVICES=0 python tools/run_train.py \
    --run_dir ../../data/clevr/attr_net_seg/outputs/trained_model/${TYPE} \
    --obj_ann_path /home/xingrui/vqa/ns-vqa/data/clevr/attr_net/outputs/clevr_alighed.json \
    --img_dir /home/xingrui/data/CLEVR_mini/images \
    --scene_path /home/xingrui/data/CLEVR_mini/CLEVR_mini_coco_anns.json  \
    --dataset clevr \
    --type ${TYPE} \
    --display_every 1 \
    --bbox_mode XYXY \
    # --resume ../../data/clevr/attr_net/outputs/trained_model/${TYPE}/checkpoint_best.pt
cd ..
