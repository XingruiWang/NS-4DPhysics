

# scp -r xingrui@10.161.159.26:/home/xingrui/physics_questions_generation/data/SuperCLEVR_physics_val_anno.json /home/xingrui/projects/superclevr-physics/physics_questions_generation/data/SuperCLEVR_physics_val_anno.json

# 1. Get nemo prediction
# 2. Physics reasoning

# python physics_reasoning/physics_properties_oracle.py \
#     --nemo_output /home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/output/superclever_nemo_1k_v3/ckpts/superclever_occ0_aeroplane_val.pth \
    # --output data/scene/superclevr_physics_scene_time_prediction.json

CUDA_VISIBLE_DEVICES=2 python physics_reasoning/physics_properties_oracle.py \
    --output data/scene/superclevr_physics_scene_prediction.json \
    --nemo_output /home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/output_old/superclever_nemo_1k_v3_physics_5.0/ckpts/superclever_occ0_aeroplane_val.pth \

# python physics_reasoning/physics_properties_oracle.py \
#     --nemo_output /home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/output/superclever_nemo_1k_v3_images/ckpts/superclever_occ0_aeroplane_val.pth \
#     --output data/scene/superclevr_physics_scene_image_prediction.json

# ## paper setting
# python physics_reasoning/physics_properties_oracle.py \
#     --nemo_output /home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/output/superclever_nemo_1k_v3_physics_prior/ckpts/superclever_occ0_aeroplane_val-ttt.pth \
#     --output data/scene/superclevr_physics_scene_prediction.json


# 1. Factual

QUESTION=factual
# # mkdir -p data/preprocess

# publishd dataset path: /home/xingrui/ccvl15_old/Dataset/DynSuperCLEVR

python tools/preprocess_questions_superclevr.py \
    --input_questions_json /home/xingrui/projects/superclevr-physics/physics_questions_generation/output/val/questions_physics_${QUESTION}.json \
    --output_h5_file data/preprocess/superclevr_physics_questions_${QUESTION}.h5 \
    --output_vocab_json data/preprocess/superclevr_physics_vocab.json

python tools/run_test.py \
    --superclevr_question_path data/preprocess/superclevr_physics_questions_${QUESTION}.h5 \
    --superclevr_scene_path data/scene/superclevr_physics_scene_prediction.json \
    --superclevr_vocab_path data/preprocess/superclevr_physics_vocab.json \
    --superclevr_gt_question_path /home/xingrui/projects/superclevr-physics/physics_questions_generation/output/val/questions_physics_${QUESTION}.json \
    --save_result_path data/reason/results_${QUESTION}.json \
    --length 10000 

