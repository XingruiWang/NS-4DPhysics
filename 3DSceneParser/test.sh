CUDA_VISIBLE_DEVICES=1,2,3,4 python3 scripts/demo.py \
    --cate car \
    --config config/omni_nemo_pose_3d.yaml \
    --save_dir exp/pose_estimation_test \
    --checkpoint exp/pose_estimation_3d_nemo_car/ckpts/model_800.pth