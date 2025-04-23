# Make the initial prediction of the scene for resimulation
baselineName=images
# python physics_reasoning/simulation.py \
#     --nemo_output data/scene/superclevr_physics_scene_time_prediction.json \
#     --output_folder /home/xingrui/projects/superclevr-physics/physical_symbolic/data/time/resimulate_counterfactual
# mkdir -p data/scene/baseline_${baselineName}
# python physics_reasoning/process_scene_files.py \
#     --data_dir /ccvl/net/ccvl15/xingrui/baseline_${baselineName}/resimulate_counterfactual_all/ \
#     --output_val data/scene/baseline_${baselineName}/SuperCLEVR_physics_val_counterfactual.json \
#     --train_scene_length 1000 \
#     --val_scene_length 100


QUESTION=predictive
# # # mkdir -p data/preprocess
# python tools/preprocess_questions_superclevr.py \
#     --input_questions_json /ccvl/net/ccvl15/xingrui/physics_questions_generation/output/val/questions_physics_${QUESTION}.json \
#     --output_h5_file data/preprocess/superclevr_physics_questions_${QUESTION}.h5 \
#     --output_vocab_json data/preprocess/superclevr_physics_vocab_${QUESTION}.json

python tools/run_test.py \
    --superclevr_question_path data/preprocess/superclevr_physics_questions_${QUESTION}.h5 \
    --superclevr_scene_path data/scene/baseline_${baselineName}/SuperCLEVR_physics_val_${QUESTION}.json \
    --superclevr_vocab_path data/preprocess/superclevr_physics_vocab_${QUESTION}.json \
    --superclevr_gt_question_path /home/xingrui/projects/superclevr-physics/physics_questions_generation/output/val/questions_physics_${QUESTION}.json \
    --save_result_path data/reason/results_${QUESTION}.json \
    --length 10000 

# 2. Counterfactual
# python physics_reasoning/counterfactual_simulation.py \
#     --nemo_output /home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/output/superclever_nemo_1k_v3/ckpts/superclever_occ0_aeroplane_val.pth \
#     --output data/scene/superclevr_physics_scene_prediction.json