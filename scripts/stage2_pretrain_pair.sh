python stage2_pair.py --filename stage2_pair/v3 --mode pretrain --warmup_steps 1000 --stage1_path ./all_checkpoints/stage1_pair/v10/stage1.ckpt --bert_name selfiesbert

python stage2_pair.py --filename stage2_pair_ft/v3 --mode ft --warmup_steps 200 --stage2_path ./all_checkpoints/stage2_pair/v2/last.ckpt --root ../Datasets/3D-MoIT/crossdocked/ --inference_batch_size 8 --bert_name selfiesbert
