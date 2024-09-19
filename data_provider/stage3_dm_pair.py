# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from data_provider.pair_dataset import PairDataset as MolDataset
from torch.utils.data import DataLoader
from data_provider.unimol_dataset import D3Collater, HoloCollater
from torch_geometric.loader.dataloader import Collater
from unicore.data import Dictionary


class MyCollater:
    def __init__(self, tokenizer, text_max_len, pad_idx, load_3d=False):
        self.pad_idx = pad_idx
        self.load_3d = load_3d
        self.d3_collater = D3Collater(pad_idx)
        self.holo_collater = HoloCollater(pad_idx)
        self.d2_collater = Collater([], [])
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def __call__(self, batch):
        if self.load_3d:
            pkt_batch, holo_batch, mol_batch, mol_batch_enc = zip(*batch)
            # padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(pkt_batch)

            pocket_padded_atom_vec, pocket_padded_coordinates, pocket_padded_edge_type, pocket_padded_dist, pocket_name = self.d3_collater(
                pkt_batch)
            mol_padded_atom_vec, mol_padded_coordinates, mol_padded_edge_type, mol_padded_dist, smiles = self.d3_collater(
                mol_batch)
            holo_padded_coordinates, holo_padded_pocket_coordinates, padded_distance, holo_padded_distance, holo_center = self.holo_collater(
                holo_batch)
            mol_padded_atom_vec_enc, _, mol_padded_edge_type_enc, _, rdmol = self.d3_collater(mol_batch_enc)

            # return (padded_atom_vec, padded_dist, padded_edge_type), text_tokens
            return (pocket_padded_atom_vec, pocket_padded_dist, pocket_padded_edge_type), (
            holo_padded_coordinates, holo_padded_pocket_coordinates, padded_distance, holo_padded_distance,
            holo_center), (mol_padded_atom_vec, mol_padded_dist, mol_padded_edge_type, pocket_name, smiles), (
            mol_padded_atom_vec_enc, mol_padded_dist, mol_padded_edge_type_enc)
        else:
            return self.d2_collater(batch)


class Stage3DM(LightningDataModule):
    def __init__(
            self,
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = 'data/',
            text_max_len: int = 128,
            pkt_dictionary=None,
            mol_dictionary=None,
            tokenizer=None,
            args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        # self.dictionary = dictionary
        self.pkt_dictionary = pkt_dictionary
        self.mol_dictionary = mol_dictionary
        self.tokenizer = tokenizer
        self.args = args
        if args.mode == 'pretrain':
            self.train_dataset = MolDataset(root + '/pretrain/', text_max_len, pkt_dictionary, mol_dictionary,
                                            args.unimol_max_atoms,
                                            enriched_descrption=args.enriched_descrption).shuffle()
        else:
            self.train_dataset = MolDataset(root + '/train/', text_max_len, pkt_dictionary, mol_dictionary,
                                            args.unimol_max_atoms).shuffle()
        self.val_dataset = MolDataset(root + '/test/', text_max_len, pkt_dictionary, mol_dictionary,
                                      args.unimol_max_atoms).shuffle()
        self.test_dataset = MolDataset(root + '/test/', text_max_len, pkt_dictionary, mol_dictionary,
                                       args.unimol_max_atoms)
        self.val_dataset_match = MolDataset(root + '/test/', text_max_len, pkt_dictionary, mol_dictionary,
                                            args.unimol_max_atoms).shuffle()
        self.test_dataset_match = MolDataset(root + '/test/', text_max_len, pkt_dictionary, mol_dictionary,
                                             args.unimol_max_atoms).shuffle()

        self.val_match_loader = DataLoader(self.val_dataset_match,
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=False,
                                           drop_last=False,
                                           persistent_workers=True,
                                           collate_fn=MyCollater(tokenizer, self.args.text_max_len,
                                                                 self.pkt_dictionary.pad(), self.args.use_3d))
        self.test_match_loader = DataLoader(self.test_dataset_match,
                                            batch_size=self.match_batch_size,
                                            shuffle=False,
                                            num_workers=self.num_workers,
                                            pin_memory=False,
                                            drop_last=False,
                                            persistent_workers=True,
                                            collate_fn=MyCollater(tokenizer, self.args.text_max_len,
                                                                  self.pkt_dictionary.pad(), self.args.use_3d))

    def load_unimol_dict(self):
        mol_dictionary = Dictionary.load('./data_provider/mol_dict.txt')
        pkt_dictionary = Dictionary.load('./data_provider/pkt_dict.txt')
        mol_dictionary.add_symbol("[MASK]", is_special=True)
        pkt_dictionary.add_symbol("[MASK]", is_special=True)
        return pkt_dictionary, mol_dictionary
        # dictionary.add_symbol("[MASK]", is_special=True)
        # return dictionary

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.pkt_dictionary.pad(), self.args.use_3d)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.match_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.pkt_dictionary.pad(), self.args.use_3d)
        )

        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.match_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.pkt_dictionary.pad(), self.args.use_3d)
        )

        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--match_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str,
                            default='./')
        parser.add_argument('--text_max_len', type=int, default=256)
        return parent_parser
