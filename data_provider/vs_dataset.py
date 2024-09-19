import os

import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset

from data_provider.unimol_dataset import LMDBDataset_cid

class SmiDataset_cid(Dataset):
    def __init__(self, path, dictionary, max_atoms=256, mol_dict=None):
        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()
        self.mol_dict = mol_dict
        if self.mol_dict is not None:
            self.mol_num_types = len(mol_dict)
            self.mol_bos = mol_dict.bos()
            self.mol_eos = mol_dict.eos()
        self.lmdb_dataset = LMDBDataset_cid(path)

        self.max_atoms = max_atoms
        ## the following is the default setting of uni-mol's pretrained weights
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

    def __len__(self):
        return len(self.lmdb_dataset)

    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    def __getitem__(self, cid):

        data = self.lmdb_dataset[cid]
        # mol = data['smi']
        # selfies = to_selfies(mol)
        if self.mol_dict is not None:
            mol_atoms = data['atoms']  # mol_atoms is atom list in the 3d-m
            mol_atoms = mol_atoms.copy()  # o
            mol_atoms = np.array(mol_atoms)
            mol_coordinate_set = data['coordinates']
            size = len(mol_coordinate_set)
            sample_idx = np.random.randint(size)
            mol_coordinates = mol_coordinate_set[sample_idx].astype(np.float32)

            assert len(mol_atoms) == len(mol_coordinates) and len(mol_atoms) > 0
            assert mol_coordinates.shape[1] == 3

            if self.remove_hydrogen:
                mask_hydrogen = mol_atoms != "H"
                if sum(mask_hydrogen) > 0:
                    mol_atoms = mol_atoms[mask_hydrogen]
                    mol_coordinates = mol_coordinates[mask_hydrogen]

            if not self.remove_hydrogen and self.remove_polar_hydrogen:
                end_idx = 0
                for i, atom in enumerate(mol_atoms[::-1]):
                    if atom != "H":
                        break
                    else:
                        end_idx = i + 1
                if end_idx != 0:
                    mol_atoms = mol_atoms[:-end_idx]
                    mol_coordinates = mol_coordinates[:-end_idx]

            if len(mol_atoms) > self.max_atoms:
                index = np.random.permutation(len(mol_atoms))[:self.max_atoms]
                mol_atoms = mol_atoms[index]
                mol_coordinates = mol_coordinates[index]

            mol_atom_vec = torch.from_numpy(self.mol_dict.vec_index(mol_atoms)).long()

            if self.normalize_coords:
                mol_coordinates = mol_coordinates - mol_coordinates.mean(axis=0)

            if self.add_special_token:
                mol_atom_vec = torch.cat(
                    [torch.LongTensor([self.mol_bos]), mol_atom_vec, torch.LongTensor([self.mol_eos])])
                mol_coordinates = np.concatenate([np.zeros((1, 3)), mol_coordinates, np.zeros((1, 3))], axis=0)

            mol_edge_type = mol_atom_vec.view(-1, 1) * self.mol_num_types + mol_atom_vec.view(1, -1)
            mol_dist = distance_matrix(mol_coordinates, mol_coordinates).astype(np.float32)
            mol_coordinates, mol_dist = torch.from_numpy(mol_coordinates), torch.from_numpy(mol_dist)
            mol = (mol_atom_vec, mol_coordinates, mol_edge_type, mol_dist, mol_atoms)
        label = data['label']

        return mol, label, 1



class D3Dataset_cid(Dataset):
    def __init__(self, path, dictionary, max_atoms=256, vs=False, mol_dict=None):
        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()
        self.vs = vs
        self.mol_dict = mol_dict
        if self.mol_dict is not None:
            self.mol_num_types = len(mol_dict)
            self.mol_bos = mol_dict.bos()
            self.mol_eos = mol_dict.eos()

        self.lmdb_dataset = LMDBDataset_cid(path)

        self.max_atoms = max_atoms
        ## the following is the default setting of uni-mol's pretrained weights
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

    def __len__(self):
        return len(self.lmdb_dataset)

    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    def __getitem__(self, cid):

        data = self.lmdb_dataset[cid]
        if not self.vs:
            mol = data['smi']
            if self.mol_dict is not None:
                mol_atoms = data['atoms']  # mol_atoms is atom list in the 3d-m
                mol_atoms = mol_atoms.copy()  # o
                mol_atoms = np.array(mol_atoms)
                mol_coordinate_set = data['coordinates']
                size = len(mol_coordinate_set)
                sample_idx = np.random.randint(size)
                mol_coordinates = mol_coordinate_set[sample_idx].astype(np.float32)

                assert len(mol_atoms) == len(mol_coordinates) and len(mol_atoms) > 0
                assert mol_coordinates.shape[1] == 3

                if self.remove_hydrogen:
                    mask_hydrogen = mol_atoms != "H"
                    if sum(mask_hydrogen) > 0:
                        mol_atoms = mol_atoms[mask_hydrogen]
                        mol_coordinates = mol_coordinates[mask_hydrogen]

                if not self.remove_hydrogen and self.remove_polar_hydrogen:
                    end_idx = 0
                    for i, atom in enumerate(mol_atoms[::-1]):
                        if atom != "H":
                            break
                        else:
                            end_idx = i + 1
                    if end_idx != 0:
                        mol_atoms = mol_atoms[:-end_idx]
                        mol_coordinates = mol_coordinates[:-end_idx]

                if len(mol_atoms) > self.max_atoms:
                    index = np.random.permutation(len(mol_atoms))[:self.max_atoms]
                    mol_atoms = mol_atoms[index]
                    mol_coordinates = mol_coordinates[index]

                mol_atom_vec = torch.from_numpy(self.mol_dict.vec_index(mol_atoms)).long()

                if self.normalize_coords:
                    mol_coordinates = mol_coordinates - mol_coordinates.mean(axis=0)

                if self.add_special_token:
                    mol_atom_vec = torch.cat(
                        [torch.LongTensor([self.mol_bos]), mol_atom_vec, torch.LongTensor([self.mol_eos])])
                    mol_coordinates = np.concatenate([np.zeros((1, 3)), mol_coordinates, np.zeros((1, 3))], axis=0)

                mol_edge_type = mol_atom_vec.view(-1, 1) * self.mol_num_types + mol_atom_vec.view(1, -1)
                mol_dist = distance_matrix(mol_coordinates, mol_coordinates).astype(np.float32)
                mol_coordinates, mol_dist = torch.from_numpy(mol_coordinates), torch.from_numpy(mol_dist)
                mol = (mol_atom_vec, mol_coordinates, mol_edge_type, mol_dist, data['smi'])
            description = to_selfies(data['smi'])
            pocket = data['pocket']
            enriched_description = f"The molecules that bind to protein pocket {pocket} could be " + description

        atoms_orig = np.array(data['pocket_atoms'])
        atoms = atoms_orig.copy()
        atoms = np.array([self.pocket_atom(atom) for atom in atoms])

        coordinate_set = data['pocket_coordinates']
        coordinates = np.array(coordinate_set).astype(np.float32)
        coordinates = coordinates[0] if len(coordinate_set) == 1 else coordinates
        assert len(atoms) == len(coordinates) and len(atoms) > 0
        assert coordinates.shape[1] == 3

        ## deal with the hydrogen
        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            if sum(mask_hydrogen) > 0:
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]

        ## deal with cropping
        if len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        assert 0 < len(atoms) <= self.__max_atoms

        atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)
        if self.vs:
            return atom_vec, coordinates, edge_type, dist
        return atom_vec, coordinates, edge_type, dist, mol, description, enriched_description

class VsDataset(Dataset):
    def __init__(self, root, text_max_len, unimol_dict=None, max_atoms=256, prompt='', return_prompt=False,
                 enriched_descrption=False, data_type='mol', mol_dict=None):
        super(VsDataset, self).__init__()
        self.prompt = prompt
        self.return_prompt = return_prompt
        self.enriched_descrption = enriched_descrption

        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        self.data_type = data_type

        # target_path = os.path.join(root, 'data.lmdb')
        mol_target_path = os.path.join(root, 'mols_new.lmdb') if os.path.exists(os.path.join(root, 'mols_new.lmdb')) else os.path.join(root, 'mols.lmdb')
        pkt_target_path = (
            os.path.join(root, 'pocket_new.lmdb') if os.path.exists(os.path.join(root, 'pocket_new.lmdb'))
            else os.path.join(root, 'pockets.lmdb') if os.path.exists(os.path.join(root, 'pockets.lmdb'))
            else os.path.join(root, 'pocket.lmdb')
        )
        self.mol_dataset = SmiDataset_cid(mol_target_path, unimol_dict, max_atoms, mol_dict=mol_dict)
        self.d3_dataset = D3Dataset_cid(pkt_target_path, unimol_dict, max_atoms, vs=True)
        if self.data_type == 'mol':
            self.cid_list = [cid.decode() for cid in self.mol_dataset.lmdb_dataset._keys]
        else:
            self.cid_list = [cid.decode() for cid in self.d3_dataset.lmdb_dataset._keys]
        self.permutation = None

    def shuffle(self):
        ## shuffle the dataset using a permutation matrix
        self.permutation = torch.randperm(len(self)).numpy()
        return self

    def __len__(self):
        return len(self.cid_list)

    def get_3d(self, index):
        if self.data_type == 'mol':
            mol, label, selfies = self.mol_dataset[self.cid_list[index]]
            atom_vec, coordinates, edge_type, dist = self.d3_dataset[self.cid_list[0]]
        else:
            atom_vec, coordinates, edge_type, dist = self.d3_dataset[self.cid_list[index]]
            mol, label, selfies = self.mol_dataset[self.cid_list[0]]

        description = selfies
        enriched_description = selfies

        if self.enriched_descrption:
            return_description = enriched_description
        else:
            return_description = description

        if self.return_prompt:
            smiles_prompt = self.prompt.format(selfies[:96])
            return (atom_vec, coordinates, edge_type, dist, mol), label, smiles_prompt, return_description, index
        return (atom_vec, coordinates, edge_type, dist, mol), label, return_description

    def __getitem__(self, index):
        ## consider the permutation
        if self.permutation is not None:
            index = self.permutation[index]
        return self.get_3d(index)

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask


if __name__ == '__main__':
    from unicore.data import Dictionary

    dictionary = Dictionary.load('data_provider/pkt_dict.txt')
    dictionary.add_symbol("[MASK]", is_special=True)
    dataset = PktDataset('../Datasets/Drug/clip/train_no_test_af' + '/pretrain/',
                         128, dictionary, 256)

    # 初始化计数器和统计量
    total_files = 0
    word_counts = []

    for index in range(0, len(dataset)):
        total_files += 1
        (atom_vec, coordinates, edge_type, dist, smiles), description = dataset[index]
        word_count = len(description)
        word_counts.append(word_count)

    # 计算统计量
    average_words = sum(word_counts) / total_files if total_files else 0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    import IPython;

    IPython.embed()
    # 输出结果
    print(f"Total files: {total_files}")
    print(f"Average words per file: {average_words:.2f}")
    print(f"Minimum words in a file: {min_words}")
    print(f"Maximum words in a file: {max_words}")
