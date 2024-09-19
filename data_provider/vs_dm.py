# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater

from data_provider.unimol_dataset import D3Collater
from data_provider.vs_dataset import VsDataset


class MyCollater:
    def __init__(self, text_max_len, pad_idx, load_3d=False):
        self.pad_idx = pad_idx
        self.load_3d = load_3d
        self.d3_collater = D3Collater(pad_idx)
        self.d2_collater = Collater([], [])
        # self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def __call__(self, batch):
        if self.load_3d:
            d3_batch, label_batch, text_batch = zip(*batch)
            padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, mol = self.d3_collater(d3_batch)
            mol_padded_atom_vec, mol_padded_coordinates, mol_padded_edge_type, mol_padded_dist, protein_name = self.d3_collater(mol)
            return (padded_atom_vec, padded_dist, padded_edge_type, (mol_padded_atom_vec, mol_padded_dist, mol_padded_edge_type)), mol, label_batch
        else:
            return self.d2_collater(batch)


class VsDM(LightningDataModule):
    def __init__(
            self,
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = 'data',
            text_max_len: int = 128,
            dictionary=None,
            # tokenizer=None,
            mol_dict=None,
            args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.dictionary = dictionary
        # self.tokenizer = tokenizer
        self.args = args
        self.val_dataset_smi = VsDataset(root, text_max_len, dictionary, args.unimol_max_atoms,
                                         data_type='mol',mol_dict=mol_dict)
        self.val_dataset_pkt = VsDataset(root, text_max_len, dictionary, args.unimol_max_atoms,
                                         data_type='pkt',mol_dict=mol_dict)
        self.val_dataset_match_smi = VsDataset(root, text_max_len, dictionary, args.unimol_max_atoms,
                                               data_type='mol',mol_dict=mol_dict)
        self.val_dataset_match_pkt = VsDataset(root, text_max_len, dictionary, args.unimol_max_atoms,
                                               data_type='pkt',mol_dict=mol_dict)

        self.val_match_loader_smi = DataLoader(self.val_dataset_match_smi,
                                               batch_size=self.match_batch_size,
                                               shuffle=False,
                                               num_workers=self.num_workers,
                                               pin_memory=False,
                                               drop_last=False,
                                               persistent_workers=True,
                                               collate_fn=MyCollater(self.args.text_max_len,
                                                                     self.dictionary.pad(), self.args.use_3d))
        self.val_match_loader_pkt = DataLoader(self.val_dataset_match_pkt,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=self.num_workers,
                                               pin_memory=False,
                                               drop_last=False,
                                               persistent_workers=True,
                                               collate_fn=MyCollater(self.args.text_max_len,
                                                                     self.dictionary.pad(), self.args.use_3d))

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.args.text_max_len, self.dictionary.pad(), self.args.use_3d)
        )

        return loader

    def val_dataloader_smi(self):
        loader = DataLoader(
            self.val_dataset_smi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.args.text_max_len, self.dictionary.pad(), self.args.use_3d)
        )

        return loader

    def val_dataloader_pkt(self):
        loader = DataLoader(
            self.val_dataset_pkt,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.args.text_max_len, self.dictionary.pad(), self.args.use_3d)
        )

        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=2048)
        parser.add_argument('--match_batch_size', type=int, default=2048)
        parser.add_argument('--root', type=str,
                            default='./data')
        parser.add_argument('--text_max_len', type=int, default=256)
        return parent_parser
