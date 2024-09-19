import csv
import os
import pickle
from typing import Any, Dict
import torch
from model.dlip_dock import Blip2Qformer
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class Blip2Stage2(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False

        self.blip2qformer = Blip2Qformer(args.gtm, args.lm, args.bert_name, args.temperature, args.gin_num_layers,
                                     args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token,
                                     args.cross_attention_freq, args.projection_dim, args)
        self.enable_flash = args.enable_flash
        self.save_hyperparameters(args)

        self.test_step_outputs = []

    def load_from_stage1_checkpoint(self, path):
        print('loading from stage1 checkpoint')
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        state_dict = {k.split('blip2qformer.')[1]: v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = self.blip2qformer.load_state_dict(state_dict, strict=False)
        print('missing keys')
        print(missing_keys)
        print('unexpected keys')
        print(unexpected_keys)
        return self

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr,
                                                           self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr,
                                                         self.args.init_lr, self.args.lr_decay_rate,
                                                         self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_validation_epoch_end(self):
        if self.enable_flash:
            replace_llama_attn_with_flash_attn()

    # def on_train_epoch_start(self) -> None:
    #     if self.enable_flash:
    #         replace_flash_attn_with_llama_attn()

    def on_validation_epoch_start(self) -> None:
        if self.enable_flash:
            replace_flash_attn_with_llama_attn()

    def on_test_epoch_start(self) -> None:
        if self.enable_flash:
            replace_flash_attn_with_llama_attn()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def save_predictions(self, predictions, targets, names):
        assert len(predictions) / self.args.num_captions == len(targets)
        with open(os.path.join(self.logger.log_dir, 'txt'), 'w', encoding='utf8') as f:
            # for p, t in zip(predictions, targets):
            #     line = {'prediction': p, 'target': t}
            #     f.write(json.dumps(line, ensure_ascii=True) + '\n')
            for i in range(len(targets)):
                for j in range(self.args.num_captions):
                    line = {'name': names[i], 'prediction': predictions[i * self.args.num_captions + j],
                            'target': targets[i]}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[0][0].size(0)
        ###============== Overall Loss ===================###
        loss, cross_loss, holo_loss, holo_distance_predict, cross_distance_predict = self.blip2qformer(batch)
        self.log("train_loss", float(loss), batch_size=batch_size, sync_dist=True)
        self.log("train_cross_loss", float(cross_loss), batch_size=batch_size, sync_dist=True)
        self.log("train_holo_loss", float(holo_loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        batch_size = batch[0][0].size(0)
        loss, cross_loss, holo_loss, holo_distance_predict, cross_distance_predict = self.blip2qformer(batch)
        ###============== Overall Loss ===================###
        self.log("val_loss", float(loss), batch_size=batch_size, sync_dist=True)
        self.log("val_cross_loss", float(cross_loss), batch_size=batch_size, sync_dist=True)
        self.log("val_holo_loss", float(holo_loss), batch_size=batch_size, sync_dist=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graph_batch, holo_batch, mol_batch, mol_batch_enc = batch
        # import IPython
        # IPython.embed()
        ###============== Captioning Results ===================###
        loss, cross_loss, holo_loss, holo_distance_predict, cross_distance_predict, score_label = self.blip2qformer(batch)
        logging_output = {}
        # logging_output['smi_name'] = list(pickle.loads(mol_batch[-1].cpu().numpy().tobytes()))
        logging_output['pocket_name'] = list(pickle.loads(mol_batch[-2].cpu().numpy().tobytes()))
        logging_output["cross_distance_predict"] = cross_distance_predict.data.detach().cpu()
        logging_output['holo_distance_predict'] = holo_distance_predict.data.detach().cpu()
        logging_output['atoms'] = mol_batch[0].data.detach().cpu()
        logging_output['pocket_atoms'] = graph_batch[0].data.detach().cpu()
        logging_output['holo_center_coordinates'] = torch.cat(holo_batch[-1]).reshape(-1,3).data.detach().cpu()
        logging_output['holo_coordinates'] = holo_batch[0].data.detach().cpu()
        logging_output['pocket_coordinates'] = holo_batch[1].data.detach().cpu()
        logging_output['loss'] = loss.data.detach().cpu()
        logging_output['cross_loss'] = cross_loss.data.detach().cpu()
        logging_output['holo_loss'] = holo_loss.data.detach().cpu()
        logging_output['score_label'] = score_label.data.detach().cpu()

        self.test_step_outputs.append(logging_output)
        return loss

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        if self.global_rank == 0:
            save_path = os.path.join(self.logger.log_dir, 'test_results.pkl')
            pickle.dump(outputs, open(save_path, "wb"))

            csv_file_path = os.path.join(self.logger.log_dir, 'valid_results.csv')
            with open(csv_file_path, mode='w', newline='') as csv_file:
                # 创建 CSV 写入器
                writer = csv.writer(csv_file)

                # 写入表头
                writer.writerow(['pocket_name', 'loss', 'cross_loss', 'holo_loss', 'score_label'])
                for logging_output in outputs:
                    for pocket_name, loss, cross_loss, holo_loss, score_label in zip(logging_output['pocket_name'], logging_output['loss'], logging_output['cross_loss'], logging_output['holo_loss'], logging_output['score_label']):
                        writer.writerow([pocket_name, loss.item(), cross_loss.item(), holo_loss.item(), score_label.item()])


    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='selfiesbert')
        parser.add_argument('--projection_dim', type=int, default=256)
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        
        # OPT
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)
        parser.add_argument('--use_mol_3d', action='store_true', default=False)

        parser.add_argument('--save_every_n_epochs', type=int, default=1)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-8, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=1)
        return parent_parser


