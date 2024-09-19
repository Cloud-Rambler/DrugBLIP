python stage3_pair.py --filename stage3_pair_ft/v3 --mode ft --warmup_steps 200 --init_checkpoint ./all_checkpoints/stage2_pair_ft/3dblip/v2_test_v1/epoch=99.ckpt --match_batch_size 4


python stage3_pair.py --filename stage3_pair_eval/v0 --mode eval --warmup_steps 200 --init_checkpoint ./all_checkpoints/stage3_pair_ft/v0/last.ckpt --match_batch_size 4 --root ../Datasets/3D-MoIT/unimol/docking_score/benchmark --devices 0

