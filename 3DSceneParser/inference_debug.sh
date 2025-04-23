
# SIGMA=3
# CUDA_VISIBLE_DEVICES=6 python scripts/inference.py \
#     --config config/superclevr_nemo_joint_3d.yaml \
#     --save_dir output_old/superclever_nemo_1k_v3_physics_resnet50_${SIGMA}/ckpts \
#     --checkpoint exp/superclever_nemo/ckpts/model_135.pth \
#     --sigma ${SIGMA} \
#     --physics_prior \


    
        # --checkpoint exp/superclever_nemo_1k_resnet50/ckpts/last_checkpoints.pth \


    # --save_dir output_old/superclever_nemo_1k_v3_sigma_${SIGMA}/ckpts \


    # --save_dir exp/superclever_nemo_1k/ckpts \

    # --checkpoint exp/superclever_nemo/ckpts/model_135.pth

    # --checkpoint exp/superclever_nemo_1k/ckpts/last_checkpoints.pth

    # --checkpoint exp/superclever_nemo/ckpts/model_150.pth
    # --checkpoint exp/superclever_nemo_1k/ckpts/model_120.pth
    # --checkpoint exp/superclever_nemo/ckpts/ckpts_train_100/model_150.pth
    # --checkpoint exp/superclever_nemo/ckpts/ckpts/synthetic/model_1000.pth
    # --checkpoint exp/superclever_nemo/ckpts/model_10.pth
    # /home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/exp/superclever_nemo/ckpts/model_135.pth

# SIGMA=3
# CUDA_VISIBLE_DEVICES=6 python scripts/inference.py \
#     --config config/superclevr_nemo_joint_3d.yaml \
#     --save_dir output_old/superclever_nemo_1k_v3_physics_resnet50_${SIGMA}/ckpts \
#     --checkpoint exp/superclever_nemo/ckpts/model_135.pth \
#     --sigma ${SIGMA} \
#     --physics_prior \
#     --vis_inference

SIGMA=3
CUDA_VISIBLE_DEVICES=6 python scripts/demo.py \
    --config config/superclevr_nemo_joint_3d.yaml \
    --save_dir output_old/superclever_nemo_1k_v3_physics_resnet50_${SIGMA}/ckpts \
    --checkpoint exp/superclever_nemo/ckpts/model_135.pth \
    --sigma ${SIGMA} \
    --physics_prior \
    --vis_inference