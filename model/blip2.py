"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os

import torch
import torch.nn as nn

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from unicore.data import Dictionary
from model.unimol import SimpleUniMolModel
    
class Blip2Base(BaseModel):

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_unimol_encoder(cls, args):
        mol_dictionary = Dictionary.load('./data_provider/mol_dict.txt')
        mol_dictionary.add_symbol("[MASK]", is_special=True)
        # mol_dictionary.specials.add('Au')
        mol_unimol_model = SimpleUniMolModel(args, mol_dictionary)
        ckpt = torch.load('/mnt/ai4x_ceph/fandiwu/buddy1/wangrubo/ckpt/3D-MoLM/uni-mol/mol_pre_no_h_220816.pt', map_location=torch.device('cpu'))['model']
        missing_keys, unexpected_keys = mol_unimol_model.load_state_dict(ckpt, strict=False)

        pkt_dictionary = Dictionary.load('./data_provider/pkt_dict.txt')
        pkt_dictionary.add_symbol("[MASK]", is_special=True)
        pkt_unimol_model = SimpleUniMolModel(args, pkt_dictionary)
        ckpt = torch.load('/mnt/ai4x_ceph/fandiwu/buddy1/wangrubo/ckpt/3D-MoLM/uni-mol/pocket_pre_220816.pt', map_location=torch.device('cpu'))['model']
        missing_keys, unexpected_keys = pkt_unimol_model.load_state_dict(ckpt, strict=False)
        # if len(missing_keys) or len(unexpected_keys):
        #     print(missing_keys)
        #     print(unexpected_keys)
        
        mol_ln_graph = nn.LayerNorm(mol_unimol_model.num_features)
        pkt_ln_graph = nn.LayerNorm(pkt_unimol_model.num_features)
        # return unimol_model, ln_graph, dictionary
        return pkt_unimol_model, pkt_ln_graph, pkt_dictionary, mol_unimol_model, mol_ln_graph, mol_dictionary


    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


