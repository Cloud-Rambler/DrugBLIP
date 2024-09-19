"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from copy import deepcopy
# from lavis.common.registry import registry
# from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput
# from lavis.common.dist_utils import is_dist_avail_and_initialized
from model.blip2 import Blip2Base
from model.dist_funs import pl_concat_all_gather
from model.transformer_encoder_with_pair import TransformerEncoderWithPair
from model.unimol import NonLinearHead, DistanceHead



class Blip2Qformer(Blip2Base):

    def __init__(
            self,
            gtm,
            lm,
            bert_name,
            temperature,
            gin_num_layers,
            gin_hidden_dim,
            gin_drop_ratio,
            tune_gnn=False,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            args=None,
    ):
        super().__init__()
        self.gtm = gtm
        self.lm = lm
        self.args = args

        self.pkt_encoder, self.ln_pkt, self.pkt_dictionary, self.mol_encoder, self.ln_mol, self.mol_dictionary = self.init_unimol_encoder(
            args)

        self.mol_num_types = len(self.mol_dictionary)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.pkt_encoder.named_parameters():
                param.requires_grad = False
            self.pkt_encoder = self.pkt_encoder.eval()
            self.pkt_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(14))
        self.graph_proj = nn.Linear(self.pkt_encoder.num_features, embed_dim)
        self.mol_proj = nn.Linear(self.pkt_encoder.num_features, embed_dim)

        self.gtm_head = nn.Linear(self.pkt_encoder.num_features, 2)


        self.cross_distance_project = NonLinearHead(
            args.unimol_encoder_embed_dim * 2 + args.unimol_encoder_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.unimol_encoder_embed_dim + args.unimol_encoder_attention_heads, "relu"
        )

        self.temperature = temperature

    def contrast(self, features_graph, features_text, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        '''
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(
            -1)).squeeze(-2)  # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1)  # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, pockets=None,
                        mols=None, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)
        if pockets is not None:

            pocket_list = [pocket.cpu().numpy().tobytes() for pocket in pockets]

            pocket_list = [list(pickle.loads(pocket)) for pocket in pocket_list]
            pocket_list = [item for sublist in pocket_list for item in sublist]
            pockets = np.array(pocket_list, dtype=str)
            pockets = np.expand_dims(pockets, 1)
            matrix1 = np.repeat(pockets, len(pockets), 1)
            matrix2 = np.repeat(np.transpose(pockets), len(pockets), 0)
            pocket_duplicate_matrix = matrix1 == matrix2
            pocket_duplicate_matrix = 1 * pocket_duplicate_matrix
            pocket_duplicate_matrix = torch.tensor(pocket_duplicate_matrix, dtype=features_text_all.dtype).cuda()

            mol_list = [mol.cpu().numpy().tobytes() for mol in mols]
            mol_list = [list(pickle.loads(mol)) for mol in mol_list]
            mol_list = [item for sublist in mol_list for item in sublist]

            mols = np.array(mol_list, dtype=str)
            mols = np.expand_dims(mols, 1)
            matrix1 = np.repeat(mols, len(mols), 1)
            matrix2 = np.repeat(np.transpose(mols), len(mols), 0)
            mol_duplicate_matrix = matrix1 == matrix2
            mol_duplicate_matrix = 1 * mol_duplicate_matrix
            mol_duplicate_matrix = torch.tensor(mol_duplicate_matrix, dtype=features_text_all.dtype).cuda()

            onehot_labels = torch.eye(features_text_all.size(0)).cuda()
            indicater_matrix = pocket_duplicate_matrix + mol_duplicate_matrix - 2 * onehot_labels

            rank = dist.get_rank()
            labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

            # cosine similarity as logits
            sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze(
                dim=-1)  # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
            sim_g2t, _ = sim_q2t.max(-1)  # shape = [B, B * num_gpus]
            sim_g2t = sim_g2t * self.logit_scale.exp().detach()
            sim_g2t = indicater_matrix[rank * bs: rank * bs + bs, :] * -1e6 + sim_g2t

            sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze(
                dim=-2)  # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
            sim_t2g, _ = sim_t2q.max(-1)

            sim_t2g = sim_t2g * self.logit_scale.exp().detach()
            sim_t2g = indicater_matrix.T[rank * bs:rank * bs + bs, :] * -1e6 + sim_t2g

            loss_graph = F.cross_entropy(sim_g2t, labels)
            loss_text = F.cross_entropy(sim_t2g, labels)
            loss = (loss_graph + loss_text) / 2

            if return_sim:
                return sim_g2t[:, rank * bs:rank * bs + bs], sim_t2g[:, rank * bs:rank * bs + bs], loss
            else:
                return loss
        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze(
            dim=-1)  # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1)  # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature

        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze(
            dim=-2)  # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank * bs:rank * bs + bs], logits_per_text[:, rank * bs:rank * bs + bs], loss
        else:
            return loss

    def docking_loss(self, holo_distance_predict, cross_distance_predict, holo_distance_target, distance_target,
                     src_tokens):

        distance_mask = distance_target.ne(0)  # 0 for padding, BOS and EOS
        # 0 is impossible in the cross distance matrix, all the relevant cross distances are kept
        if self.args.dist_threshold > 0:
            distance_mask &= (
                    distance_target < self.args.dist_threshold
            )
        distance_predict = cross_distance_predict[distance_mask]
        distance_target = distance_target[distance_mask]
        distance_loss = F.mse_loss(
            distance_predict.float(), distance_target.float(), reduction="mean"
        )

        ### holo distance loss
        token_mask = src_tokens.ne(self.mol_dictionary.pad()) & \
                     src_tokens.ne(self.mol_dictionary.eos()) & \
                     src_tokens.ne(self.mol_dictionary.bos())
        holo_distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
        holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
        holo_distance_target = holo_distance_target[
            holo_distance_mask
        ]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict_train.float(),
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
        )

        loss = distance_loss + holo_distance_loss

        return loss

    def forward(self, batch):
        ## for 3d forward
        device = self.device
        graph_batch, holo_batch, mol_batch, mol_batch_enc = batch
        _, _, holo_distance_target, distance_target, _ = holo_batch

        pkt_out = self.pkt_encoder(*graph_batch)
        batch_node = pkt_out[0]
        batch_node_pair = pkt_out[1]
        batch_padding_mask = pkt_out[2]
        pocket_list = list(mol_batch[-2])
        smi_list = list(mol_batch[-1])
        pocket_list = torch.stack(pocket_list)
        smi_list = torch.stack(smi_list)

        mol_out = self.mol_encoder(*mol_batch[:3])
        batch_mol = mol_out[0]
        batch_pair_mol = mol_out[1]
        batch_padding_mask_mol = mol_out[2]
        if not self.tune_gnn:
            batch_node = batch_node.detach()
            batch_node_pair = batch_node_pair.detach()

        batch_size = batch_node.shape[0]

        batch_node = self.ln_pkt(batch_node)

        batch_mol = self.ln_mol(batch_mol)

        mol_feats = self.mol_proj(batch_mol[:, 0, :])

        graph_feats = self.graph_proj(batch_node[:, :1, :])  # shape = [B, num_q, D]

        all_pockets, all_mols = pl_concat_all_gather(pocket_list.unsqueeze(0), padding=True), pl_concat_all_gather(
            smi_list.unsqueeze(0), padding=True)

        mol_feats, graph_feats = F.normalize(mol_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        mol_feats_all, graph_feats_all = pl_concat_all_gather(mol_feats), pl_concat_all_gather(
            graph_feats)  # shape = [B * num_gpus, D]

        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, mol_feats, graph_feats_all, mol_feats_all,
                                                          all_pockets, all_mols,
                                                          return_sim=True)

        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.gtm:
            g_emb_world = batch_node
            g_pair_world = batch_node_pair
            g_mask_world = batch_padding_mask
            mol_batch_world = mol_batch_enc

            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_pair_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb_world[neg_idx])
                graph_pair_neg.append(g_pair_world[neg_idx])
                graph_mask_neg.append(g_mask_world[neg_idx])

            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_pair_neg = torch.stack(graph_pair_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each image
            mol_batch_neg = []
            mol_batch_src_neg = []
            mol_batch_pair_neg = []
            mol_batch_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                mol_batch_src_neg.append(mol_batch_world[0][neg_idx])
                mol_batch_pair_neg.append(mol_batch_world[1][neg_idx])
                mol_batch_mask_neg.append(mol_batch_world[2][neg_idx])

            mol_batch_src_neg = torch.stack(mol_batch_src_neg, dim=0)
            mol_batch_pair_neg = torch.stack(mol_batch_pair_neg, dim=0)
            mol_batch_mask_neg = torch.stack(mol_batch_mask_neg, dim=0)

            mol_batch_neg = (mol_batch_src_neg, mol_batch_pair_neg, mol_batch_mask_neg)

            mol_batch_all = tuple([torch.cat([mol_batch_world[i], mol_batch_neg[i], mol_batch_world[i]], dim=0) for i in
                                   range(3)])  # pos, neg, pos

            graph_embeds_all = torch.cat([batch_node, graph_embeds_neg, batch_node], dim=0)  # pos, neg, pos
            graph_pair_all = torch.cat([batch_node_pair, graph_pair_neg, batch_node_pair], dim=0)
            graph_padding_mask_all = torch.cat([batch_padding_mask, graph_mask_neg, batch_padding_mask], dim=0)

            output_itm = self.mol_encoder(*mol_batch_all, graph_embeds_all, graph_pair_all, graph_padding_mask_all)

            vl_embeddings = output_itm[0][:, :1, :]  # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0).to(device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Docking ========================##
        loss_lm = 0
        if self.lm:
            mol_sz = batch_mol.size(1)
            pocket_sz = batch_node.size(1)

            concat_rep = torch.cat(
                [batch_mol, batch_node], dim=-2
            )  # [batch, mol_sz+pocket_sz, hidden_dim]
            concat_mask = torch.cat(
                [batch_padding_mask_mol, batch_padding_mask], dim=-1
            )  # [batch, mol_sz+pocket_sz]
            attn_bs = batch_pair_mol.size(0) * batch_pair_mol.size(-1)

            concat_attn_bias = torch.zeros(
                attn_bs, mol_sz + pocket_sz, mol_sz + pocket_sz
            ).type_as(
                concat_rep
            )  # [batch, mol_sz+pocket_sz, mol_sz+pocket_sz]

            concat_attn_bias[:, :mol_sz, :mol_sz] = (
                batch_pair_mol.permute(0, 3, 1, 2)
                .reshape(-1, mol_sz, mol_sz)
                .contiguous()
            )
            concat_attn_bias[:, -pocket_sz:, -pocket_sz:] = (
                batch_node_pair.permute(0, 3, 1, 2)
                .reshape(-1, pocket_sz, pocket_sz)
                .contiguous()
            )

            decoder_rep = concat_rep
            decoder_pair_rep = concat_attn_bias
            for i in range(self.args.recycling):
                decoder_outputs = self.mol_encoder.encoder(
                    decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep
                )
                decoder_rep = decoder_outputs[0]
                decoder_pair_rep = decoder_outputs[1]
                decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0
                if i != (self.args.recycling - 1):
                    decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(
                        -1, mol_sz + pocket_sz, mol_sz + pocket_sz
                    )

            mol_decoder = decoder_rep[:, :mol_sz]
            pocket_decoder = decoder_rep[:, mol_sz:]

            mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
            mol_pocket_pair_decoder_rep = (decoder_pair_rep[:, :mol_sz, mol_sz:, :]
                                           + decoder_pair_rep[:, mol_sz:, :mol_sz, :].transpose(1, 2)
                                           ) / 2.0
            mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

            cross_rep = torch.cat(
                [
                    mol_pocket_pair_decoder_rep,
                    mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                    pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
                ],
                dim=-1,
            )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

            cross_distance_predict = (
                    F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
            )  # batch, mol_sz, pocket_sz

            holo_encoder_pair_rep = torch.cat(
                [
                    mol_pair_decoder_rep,
                    mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1),
                ],
                dim=-1,
            )  # [batch, mol_sz, mol_sz, 3*hidden_size]
            holo_distance_predict = self.holo_distance_project(
                holo_encoder_pair_rep
            )  # batch, mol_sz, mol_sz

            loss_lm = self.docking_loss(holo_distance_predict, cross_distance_predict, holo_distance_target,
                                        distance_target, mol_batch[0])

        return BlipOutput(
            loss=loss_gtc + loss_gtm + loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )


    def graph_forward(self, graph):
        pkt_out = self.pkt_encoder(*graph)
        batch_node = self.ln_pkt(pkt_out[0])
        graph_feats = self.graph_proj(batch_node[:, :1, :])  # shape = [B, 1, D]
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return graph_feats, batch_node, pkt_out[1], pkt_out[2]

    def text_forward(self, text, mask):
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True)  # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats

    def mol_forward(self, mol):
        mol_out = self.mol_encoder(*mol)
        batch_mol = self.ln_mol(mol_out[0])
        mol_feats = self.mol_proj(batch_mol[:, 0, :])  # shape = [B,  D]
        mol_feats = F.normalize(mol_feats, p=2, dim=-1)
        return mol_feats, batch_mol, mol_out[2]

    def compute_gtm(self, batch_node, batch_pair, batch_mask, mol_batch):
        '''
        batch_node shape = [B, N, D]
        batch_mask shape = [B, N]
        batch_mol shape = [B, D]
        batch_mol_mask shape = [B]
        '''

        output_itm = self.mol_encoder(*mol_batch, batch_node, batch_pair, batch_mask)

        gl_embeddings = output_itm[0][:, :1, :]  # shape = [B, :1, D]
        gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1)  # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        gtm_logit = gtm_logit[:, 1]  # select the axis of the positive class
        return gtm_logit
