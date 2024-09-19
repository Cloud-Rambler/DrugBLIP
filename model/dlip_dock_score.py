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
from model.unimol import NonLinearHead, DistanceHead, ClassificationHead
from torch_scatter import scatter_mean, scatter

class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

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
        # else:
        #     self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
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

        # scoring
        # TODO: add more heads,thing which features to use
        self.score_layer = ClassificationHead(
            input_dim=self.pkt_encoder.num_features,
            inner_dim=self.pkt_encoder.num_features,
            num_classes=1,
            activation_fn="relu",
            pooler_dropout=0.1,
        )


        self.cross_distance_project = NonLinearHead(
            args.unimol_encoder_embed_dim * 2 + args.unimol_encoder_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.unimol_encoder_embed_dim + args.unimol_encoder_attention_heads, "relu"
        )

        self.temperature = temperature


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

        loss = distance_loss + 3*holo_distance_loss

        return loss,distance_loss,holo_distance_loss

    def forward(self, batch):
        ## for 3d forward
        graph_batch, holo_batch, mol_batch, mol_batch_enc = batch
        _, _, holo_distance_target, distance_target, _ = holo_batch

        pkt_out = self.pkt_encoder(*graph_batch)
        batch_node = pkt_out[0]
        batch_node_pair = pkt_out[1]
        batch_padding_mask = pkt_out[2]

        mol_out = self.mol_encoder(*mol_batch[:3])
        batch_mol = mol_out[0]
        batch_pair_mol = mol_out[1]
        batch_padding_mask_mol = mol_out[2]
        if not self.tune_gnn:
            batch_node = batch_node.detach()
            batch_node_pair = batch_node_pair.detach()


        batch_node = self.ln_pkt(batch_node)

        batch_mol = self.ln_mol(batch_mol)

        ##================= Image Captioning ========================##

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

        loss, cross_loss, holo_loss = self.docking_loss(holo_distance_predict, cross_distance_predict, holo_distance_target,
                                    distance_target, mol_batch[0])


        score_predict = self.score_layer(mol_decoder)
        shape = score_predict.shape
        score_label = mol_batch[-1].view(shape[0], shape[1]).contiguous().float()

        score_loss = F.smooth_l1_loss(score_predict, score_label)
        loss += score_loss

        return loss, cross_loss, holo_loss, holo_distance_predict, cross_distance_predict, score_loss, score_predict, score_label


    def graph_forward(self, graph):
        if self.args.use_3d:
            pkt_out = self.pkt_encoder(*graph)
        else:
            pkt_out = self.pkt_encoder(graph)
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

    def scoring(self, lig_s, lig_pos, pro_s, data, dist_threhold, batch_size):
        '''
        scoring the protein-ligand binding strength
        '''
        pi, sigma, mu, dist, c_batch, _, _ = self.mdn_layer(lig_s=lig_s, lig_pos=lig_pos, lig_batch=data['ligand'].batch,
                                                               pro_s=pro_s, pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch,
                                                               edge_index=data['ligand', 'l2l', 'ligand'].edge_index[:, data['ligand'].cov_edge_mask])
        mdn_score = self.mdn_layer.calculate_probablity(pi, sigma, mu, dist)
        mdn_score[torch.where(dist > dist_threhold)[0]] = 0.
        mdn_score = scatter(mdn_score, index=c_batch, dim=0, reduce='sum', dim_size=batch_size).float()
        return mdn_score
