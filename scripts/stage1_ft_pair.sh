python stage1_pair.py --filename stage1_pair_ft/tune_gnn/v0 --mode ft --max_epochs 10 --warmup_steps 200 --lm --init_checkpoint ./all_checkpoints/stage1_pair/tune_gnn/v0/epoch=28.ckpt


python stage1_pair.py --filename stage1_pair/3dblip/ft --mode ft --max_epochs 50 --warmup_steps 1000 --use_mol_3d --tune_gnn --batch_size 6 --match_batch_size 6 --lm --init_checkpoint ./all_checkpoints/stage1_pair/3dblip/pretrain/epoch=30.ckpt --precision 32
