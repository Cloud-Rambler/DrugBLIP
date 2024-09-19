import os

from data_provider.stage3_dm_pair import Stage3DM

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import argparse
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.blip2_stage3 import Blip2Stage3
from model.unimol import SimpleUniMolModel
from model.dist_funs import MyDeepSpeedStrategy


## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage3.load_from_checkpoint(args.init_checkpoint, device=args.devices, strict=False,args=args)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage3(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    dm = Stage3DM(args.num_workers, args.batch_size, args.root, args.text_max_len, model.blip2qformer.pkt_dictionary, model.blip2qformer.mol_dictionary, args=args)


    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="./all_checkpoints/"+args.filename+"/",
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs,
                                         save_last=True,
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = len(args.devices)
        print(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )

    if args.mode == 'pretrain':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'ft':
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="stage3_pair")

    parser.add_argument('--seed', type=int, default=0, help='random seed')


    parser.add_argument('--gtm', action='store_false', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_false', help='use language modeling or not', default=True)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--use_3d', action='store_true', default=True)
    parser.add_argument('--enriched_descrption', action='store_true', default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--save_every_n_train_steps', type=int, default=1000)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--enable_flash', action='store_true', default=False)
    parser = Blip2Stage3.add_model_specific_args(parser)  # add model args
    parser = Stage3DM.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args


if __name__ == '__main__':
    main(get_args())

