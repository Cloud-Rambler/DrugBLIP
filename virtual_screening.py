import argparse
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import unicore
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
from tqdm import tqdm

from data_provider.vs_dm import VsDM
from model.blip2_stage1 import Blip2Stage1
from model.unimol import SimpleUniMolModel

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium')  # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio * n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp >= num:
                break
    return (tp * n) / (p * fp)


def calc_re(y_true, y_score, ratio_list):
    res2 = {}

    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    return res2


def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """

    # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index = np.argsort(y_score)[::-1]
    for i in range(int(len(index) * 0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)

    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list


def test_dude(model, mol_data, device, pkt_data):
    mol_reps = []
    mol_names = []
    labels = []
    pocket_reps = []
    gtm_logits = []

    for _, sample in enumerate(tqdm(mol_data)):
        with torch.no_grad():
            sample = unicore.utils.move_to_cuda(sample)

            mol_batch = sample[0][-1][:3]
            text_rep, _, _ = model.blip2qformer.mol_forward(mol_batch)

            text_feats = text_rep


            mol_emb = text_feats.detach()
            # print(mol_emb.dtype)
            mol_reps.append(mol_emb)
            labels.extend(np.array(sample[-1]))

    for _, sample in enumerate(tqdm(pkt_data)):
        with torch.no_grad():
            sample = unicore.utils.move_to_cuda(sample)
            text_batch = sample[1]
            graph_batch = sample[0][:-1]
            mol_batch = sample[0][-1][:3]
            # text_batch = text_batch.to(device)

            # graph_rep, graph_feat, graph_mask = model.blip2qformer.graph_forward_v2(graph_batch)
            graph_rep, graph_feat, _, graph_mask = model.blip2qformer.graph_forward(graph_batch)
            graph_feats = graph_rep
            pocket_emb = graph_feats[0].detach()
            pocket_reps.append(pocket_emb.unsqueeze(0))

    mol_reps = torch.cat(mol_reps, dim=0)
    pocket_reps = torch.cat(pocket_reps, dim=0)
    # gtm_logits = np.concatenate(gtm_logits, axis=0)
    pocket_reps = pocket_reps.unsqueeze(0)  # [1, N, num_qs,,D]
    mol_reps = mol_reps.unsqueeze(-1)  # [B, D, 1]

    res = torch.einsum('bmnd,idm->bin', pocket_reps, mol_reps)  # [B, num_qs]
    res_single, _ = torch.max(res, dim=-1)
    # res_single = torch.mean(res, dim=-1)
    # res_single = torch.nn.functional.softmax(res_single.float(), dim=-1)
    res_single = res_single.cpu().numpy()
    res_single = res_single.max(axis=0)


    labels = np.array(labels)
    auc, bedroc, ef, re = cal_metrics(labels, res_single, 80.5)
    print("auc", auc, "bedroc", bedroc, "ef", ef, "re", re)
    return auc, bedroc, ef, re, res_single, labels

def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage1(args)
        ckpt = torch.load(args.init_checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        # model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage1(args)

    print('total params:', sum(p.numel() for p in model.parameters()))
    model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if args.task == 'dude':
        targets = os.listdir(args.data_path)
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list = []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        for i, target in enumerate(targets):
            if target.endswith('.txt'):
                continue
            data_path = os.path.join(args.data_path, target)
            # data
            dm = VsDM(args.num_workers, args.batch_size, data_path, args.text_max_len,
                      model.blip2qformer.pkt_dictionary,
                      mol_dict=model.blip2qformer.mol_dictionary, args=args)

            # model.val_match_loader = dm.val_match_loader
            mol_data = dm.val_match_loader_smi
            pocket_data = dm.val_match_loader_pkt

            num_data = len(mol_data)
            bsz = 64
            print(num_data // bsz)

            auc, bedroc, ef, re, res_single, labels = test_dude(model, mol_data, device, pocket_data)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])
            res_list.append(res_single)
            labels_list.append(labels)

        print("auc mean", np.mean(auc_list))
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "mean", np.mean(ef_list[key]))

        for key in re_list:
            print("re", key, "mean", np.mean(re_list[key]))

        save = {}
        save["auc"] = auc_list
        save["bedroc"] = bedroc_list
        save["ef"] = ef_list
        save["re"] = re_list
        torch.save(save,
                   "./all_checkpoints/" + args.filename + f"/{os.path.basename(args.init_checkpoint)}_result_{args.task}.pt")
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="stage1")
    parser.add_argument('--data_path', type=str, default="./DUD-E/raw/all/")

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--gtm', action='store_false', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_false', help='use language modeling or not', default=True)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--use_3d', action='store_true', default=True)
    parser.add_argument('--use_mol', action='store_true', default=False)
    parser.add_argument('--enriched_descrption', action='store_true', default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['bf16-mixed', 'bf16', '32', '16-mixed'])
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--task', type=str, default='dude', choices=['dude'])
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = VsDM.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)
