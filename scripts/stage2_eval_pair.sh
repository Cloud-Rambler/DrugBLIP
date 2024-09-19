python stage2_pair.py --filename stage2_pair_eval/ --mode eval --warmup_steps 200 --stage2_path ./all_checkpoints/stage2_pair_ft/v3/epoch=09.ckpt --root ../Datasets/3D-MoIT/crossdocked/ --inference_batch_size 8 --devices 0,1 --num_captions 5


python stage2_pair.py --filename stage2_pair_eval/casf2013dock/99 --mode eval --stage2_path ./all_checkpoints/stage2_pair_ft/3dblip/v2_test_v1/epoch=99.ckpt --root ../Datasets/Drug/other/CASF-2013/docking --match_batch_size 8

python stage2_pair.py --filename stage2_pair_eval/casf2016dock/99 --mode eval --stage2_path ./all_checkpoints/stage2_pair_ft/3dblip/v2_test_v1/epoch=99.ckpt --root ../Datasets/Drug/other/CASF-2016/docking --match_batch_size 8

python stage2_pair.py --filename stage2_pair_eval/unimol/99 --mode eval --stage2_path ./all_checkpoints/stage2_pair_ft/3dblip/v2_test_v1/epoch=99.ckpt --match_batch_size 8

CUDA_VISIBLE_DEVICES=0 python stage2_pair.py --filename stage2_pair_eval/docking/2016 --mode eval --stage2_path ./all_checkpoints/stage2_pair_ft/3dblip/v2_test_v1/epoch=99.ckpt --root ../Datasets/3D-MoIT/unimol/docking_score/benchmark/2016 --match_batch_size 8 --devices 0

CUDA_VISIBLE_DEVICES=1 python stage2_pair.py --filename stage2_pair_eval/docking/2013 --mode eval --stage2_path ./all_checkpoints/stage2_pair_ft/3dblip/v2_test_v1/epoch=99.ckpt --root ../Datasets/3D-MoIT/unimol/docking_score/benchmark/2013 --match_batch_size 8 --devices 0

../Datasets/3D-MoIT/unimol/docking_score
