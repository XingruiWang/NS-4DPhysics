CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python3 scripts/train.py \
    --config config/superclevr_nemo_joint_3d.yaml \
    --save_dir exp/superclever_nemo_1k_vits8 \
    # --resume exp/superclever_nemo_1k_v3/ckpts/last_checkpoints.pth
    # --resume exp/superclever_nemo/ckpts/model_135.pth



# "resnetext2": 4, ori

# "resnetext": 8,
# "resnext50": 32,
# "resnet50": 32,
# vits8:


# The setting for ICLR submission
# --config config/superclevr_nemo_joint_3d.yaml \
# --save_dir exp/superclever_nemo_1k_v3 \
# # --resume exp/superclever_nemo_1k_v3/ckpts/last_checkpoints.pth
# # --resume exp/superclever_nemo/ckpts/model_135.pth