import os
import numpy as np
import torch
import random
import lmdb
import pickle
from functools import lru_cache
from unicore.data import data_utils
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
import selfies as sf


def collate_cross_2d(
        values,
        pad_idx,
        left_pad=False,
        pad_to_length=None,
        pad_to_multiple=1,
):
    """Convert a list of 2d tensors into a padded 2d tensor."""
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size_h - v.size(0):, size_w - v.size(1):]
            if left_pad
            else res[i][: v.size(0), : v.size(1)],
        )
    return res


def to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except Exception as e:
        print(e)
        return smiles


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def get_closest_coordinates(coordinates, sequence, n, holo_coordinates=None):
    center = np.mean(coordinates, axis=0)  # 计算坐标的中心点
    distances = euclidean_distance(coordinates, center)
    indices = np.argsort(distances)[:n]
    closest_coordinates = coordinates[indices]
    closest_sequence = sequence[indices]
    if holo_coordinates is not None:
        holo_closest_coordinates = holo_coordinates[indices]
        return closest_coordinates, closest_sequence, holo_closest_coordinates
    return closest_coordinates, closest_sequence


class LMDBDataset_cid:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, cid):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(cid.encode())
        data = pickle.loads(datapoint_pickled)
        return data


class D3Dataset_cid(Dataset):
    def __init__(self, path, dictionary, max_atoms=256):
        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

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

    def __getitem__(self, cid):
        data = self.lmdb_dataset[cid]
        smiles = data['smiles']
        description = data['description']
        enriched_description = data['enriched_description']
        ## deal with 3d coordinates
        atoms_orig = np.array(data['atoms'])
        atoms = atoms_orig.copy()
        coordinate_set = data['coordinates']
        coordinates = random.sample(coordinate_set, 1)[0].astype(np.float32)
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
        return atom_vec, coordinates, edge_type, dist, smiles, description, enriched_description


class PMDataset_cid(Dataset):
    def __init__(self, path, pkt_dictionary, mol_dictionary, max_atoms=256):
        self.pkt_dictionary = pkt_dictionary
        self.pkt_num_types = len(pkt_dictionary)
        self.bos = pkt_dictionary.bos()
        self.eos = pkt_dictionary.eos()

        self.mol_dictionary = mol_dictionary
        self.enc = self.mol_dictionary.index('Au')
        self.mol_num_types = len(mol_dictionary)

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
        # ['atoms', 'coordinates', 'pocket_atoms', 'pocket_coordinates', 'mol', 'smi', 'pocket', 'residue']
        # pocket : 'pocket_atoms', 'pocket_coordinates'
        # mol : 'atoms', 'coordinates', 'smi'
        # pocket_name: 'pocket'
        # residue: 'residue'
        # rdmol: 'mol'

        # ToDO: add distance_target / holo_distance_target / holo_coord / holo_center_coordinates / holo_pocket_coordinates

        # holo_coord: rdmol.GetConformers()[0].GetPositions()
        # holo_center_coordinates: pocket_coordinates.mean(axis=0)
        # holo_pocket_coordinates: pocket_coordinates(remove H)
        # distance_target: get the distance of holo_coordinates and holo_pocket_coordinates
        # holo_distance_target: get the distance of holo_coordinates

        rdmol = data['mol'] if 'mol' in data else data['smi']
        smiles = data['smi']
        # selfies = sf.encoder(smiles)

        pocket_name = data['pocket'].split('/')[1].split('_')[0] if '/' in data['pocket'] else data['pocket']

        # mol_data
        mol_atoms = data['atoms']  # mol_atoms is atom list in the 3d-m
        mol_atoms = mol_atoms.copy()  # o
        mol_atoms = np.array(mol_atoms)
        mol_coordinate_set = data['coordinates']
        size = len(mol_coordinate_set)
        if len(mol_coordinate_set[0]) == 3:
            mol_coordinates = np.array(mol_coordinate_set).astype(np.float32)
        else:
            sample_idx = np.random.randint(size)
            mol_coordinates = mol_coordinate_set[sample_idx].astype(np.float32)

        try:
            assert len(mol_atoms) == len(mol_coordinates) and len(mol_atoms) > 0
        except:
            print(len(mol_atoms), len(mol_coordinates))
            print(mol_atoms, mol_coordinates)
        assert len(mol_atoms) == len(mol_coordinates) and len(mol_atoms) > 0
        assert mol_coordinates.shape[1] == 3

        # pocket_data
        pocket_atoms = np.array(data['pocket_atoms'])
        pocket_atoms = pocket_atoms.copy()
        pocket_atoms = np.array([self.pocket_atom(atom) for atom in pocket_atoms])
        pocket_coordinate_set = data['pocket_coordinates']
        pocket_coordinates = np.array(pocket_coordinate_set).astype(np.float32)
        pocket_coordinates = pocket_coordinates[0] if len(pocket_coordinate_set) == 1 else pocket_coordinates

        assert len(pocket_atoms) == len(pocket_coordinates) and len(pocket_atoms) > 0
        assert pocket_coordinates.shape[1] == 3

        # for holo
        holo_coordinates = data['holo_coordinates'][0] if len(data['holo_coordinates']) == 1 else data['holo_coordinates']

        if len(mol_atoms) != len(holo_coordinates):
            holo_coordinates = mol_coordinates
        holo_pocket_coordinates = data['holo_pocket_coordinates'][0] if len(data['holo_pocket_coordinates']) == 1 else data['holo_pocket_coordinates']
        # deal with mol
        ## deal with the hydrogen
        if self.remove_hydrogen:
            mask_hydrogen = mol_atoms != "H"
            if sum(mask_hydrogen) > 0:
                mol_atoms = mol_atoms[mask_hydrogen]
                mol_coordinates = mol_coordinates[mask_hydrogen]
                holo_coordinates = holo_coordinates[mask_hydrogen]

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
                holo_coordinates = holo_coordinates[:-end_idx]

        ## deal with cropping
        if len(mol_atoms) > self.max_atoms:
            # index = np.random.permutation(len(mol_atoms))[:self.max_atoms]
            # mol_atoms = mol_atoms[index]
            # mol_coordinates = mol_coordinates[index]

            mol_coordinates, mol_atoms, holo_coordinates = get_closest_coordinates(mol_coordinates, mol_atoms,
                                                                                   self.max_atoms,
                                                                                   holo_coordinates=holo_coordinates)

        assert 0 < len(mol_atoms) <= self.__max_atoms

        mol_atom_vec = torch.from_numpy(self.mol_dictionary.vec_index(mol_atoms)).long()

        # deal with pocket
        ## deal with the hydrogen
        if self.remove_hydrogen:
            mask_hydrogen = pocket_atoms != "H"
            if sum(mask_hydrogen) > 0:
                pocket_atoms = pocket_atoms[mask_hydrogen]
                pocket_coordinates = pocket_coordinates[mask_hydrogen]
                holo_pocket_coordinates = holo_pocket_coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(pocket_atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                pocket_atoms = pocket_atoms[:-end_idx]
                pocket_coordinates = pocket_coordinates[:-end_idx]
                holo_pocket_coordinates = holo_pocket_coordinates[:-end_idx]

        ## deal with cropping
        if len(pocket_atoms) > self.max_atoms:
            # index = np.random.permutation(len(pocket_atoms))[:self.max_atoms]
            # pocket_atoms = pocket_atoms[index]
            # pocket_coordinates = pocket_coordinates[index]

            pocket_coordinates, pocket_atoms, holo_pocket_coordinates = get_closest_coordinates(pocket_coordinates,
                                                                                                pocket_atoms,
                                                                                                self.max_atoms,
                                                                                                holo_coordinates=holo_pocket_coordinates)

        assert 0 < len(pocket_atoms) <= self.__max_atoms

        pocket_atom_vec = torch.from_numpy(self.pkt_dictionary.vec_index(pocket_atoms)).long()
        if self.normalize_coords:
            mol_coordinates = mol_coordinates - mol_coordinates.mean(axis=0)
            pocket_coordinates = pocket_coordinates - pocket_coordinates.mean(axis=0)

        if self.add_special_token:
            mol_atom_vec_enc = mol_atom_vec.detach()
            mol_atom_vec = torch.cat([torch.LongTensor([self.bos]), mol_atom_vec, torch.LongTensor([self.eos])])
            mol_atom_vec_enc = torch.cat([torch.LongTensor([self.enc]), mol_atom_vec_enc, torch.LongTensor([self.eos])])
            mol_coordinates = np.concatenate([np.zeros((1, 3)), mol_coordinates, np.zeros((1, 3))], axis=0)
            pocket_atom_vec = torch.cat([torch.LongTensor([self.bos]), pocket_atom_vec, torch.LongTensor([self.eos])])
            pocket_coordinates = np.concatenate([np.zeros((1, 3)), pocket_coordinates, np.zeros((1, 3))], axis=0)
            holo_coordinates = np.concatenate([np.zeros((1, 3)), holo_coordinates, np.zeros((1, 3))], axis=0)
            holo_pocket_coordinates = np.concatenate([np.zeros((1, 3)), holo_pocket_coordinates, np.zeros((1, 3))],
                                                     axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        mol_edge_type = mol_atom_vec.view(-1, 1) * self.mol_num_types + mol_atom_vec.view(1, -1)
        mol_edge_type_enc = mol_atom_vec_enc.view(-1, 1) * self.mol_num_types + mol_atom_vec_enc.view(1, -1)
        mol_dist = distance_matrix(mol_coordinates, mol_coordinates).astype(np.float32)
        mol_coordinates, mol_dist = torch.from_numpy(mol_coordinates), torch.from_numpy(mol_dist)

        ## obtain edge types; which is defined as the combination of two atom types
        pocket_edge_type = pocket_atom_vec.view(-1, 1) * self.pkt_num_types + pocket_atom_vec.view(1, -1)
        pocket_dist = distance_matrix(pocket_coordinates, pocket_coordinates).astype(np.float32)
        pocket_coordinates, pocket_dist = torch.from_numpy(pocket_coordinates), torch.from_numpy(pocket_dist)

        # obtain holo
        distance_target = distance_matrix(holo_coordinates, holo_coordinates).astype(np.float32)
        holo_distance_target = distance_matrix(holo_coordinates, holo_pocket_coordinates).astype(np.float32)
        holo_center_coordinates = holo_pocket_coordinates.mean(axis=0)
        if self.normalize_coords:
            holo_coordinates = holo_coordinates - holo_center_coordinates
            holo_pocket_coordinates = holo_pocket_coordinates - holo_center_coordinates
        # holo = (holo_coordinates, holo_pocket_coordinates, holo_distance_target, distance_target, holo_center_coordinates)
        holo = (torch.from_numpy(holo_coordinates), torch.from_numpy(holo_pocket_coordinates),
                torch.from_numpy(holo_distance_target), torch.from_numpy(distance_target),
                torch.from_numpy(holo_center_coordinates))

        return pocket_atom_vec, pocket_coordinates, pocket_edge_type, pocket_dist, rdmol, pocket_name,smiles, holo, mol_atom_vec_enc, mol_edge_type_enc, mol_atom_vec, mol_coordinates, mol_edge_type, mol_dist


class LMDBDataset_index:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(idx.encode())
        data = pickle.loads(datapoint_pickled)
        return data


class D3Dataset_index(Dataset):
    def __init__(self, path, dictionary, max_atoms=256):
        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        self.lmdb_dataset = LMDBDataset_index(path)

        self.max_atoms = max_atoms
        ## the following is the default setting of uni-mol's pretrained weights
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, index):
        data = self.lmdb_dataset[index]
        smiles = data['smi']
        ## deal with 3d coordinates
        atoms_orig = np.array(data['atoms'])
        atoms = atoms_orig.copy()
        coordinate_set = data['coordinates_list']
        coordinates = random.sample(coordinate_set, 1)[0].astype(np.float32)
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
        if self.max_atoms > 0 and len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        assert 0 < len(atoms) < self.__max_atoms, print(len(atoms), atoms_orig, index)
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
        return atom_vec, coordinates, edge_type, dist, smiles


def collate_tokens_coords(
        values,
        pad_idx,
        left_pad=False,
        pad_to_length=None,
        pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, :] if left_pad else res[i][: len(v), :])
    return res


class D3Collater:
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, samples):
        atom_vec, coordinates, edge_type, dist, smiles = zip(*samples)
        if type(smiles[0]) is str:
            serialized_strings = pickle.dumps(smiles)
            # 将字节串转换为tensor
            smiles = torch.ByteTensor(list(serialized_strings))
        elif type(smiles[0]) is float or type(smiles[0]) is int:
            smiles = torch.tensor(smiles)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False,
                                                    pad_to_multiple=self.pad_to_multiple)  # shape = [batch_size, max_atoms]
        padded_coordinates = collate_tokens_coords(coordinates, 0, left_pad=False,
                                                   pad_to_multiple=self.pad_to_multiple)  # shape = [batch_size, max_atoms, 3]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False,
                                                        pad_to_multiple=self.pad_to_multiple)  # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False,
                                                   pad_to_multiple=self.pad_to_multiple)  # shape = [batch_size, max_atoms, max_atoms]
        return padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles


class HoloCollater:
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, samples):
        holo_coordinates, holo_pocket_coordinates, holo_distance, distance, holo_center = zip(*samples)

        padded_holo_coordinates = collate_tokens_coords(holo_coordinates, 0, left_pad=False,
                                                        pad_to_multiple=self.pad_to_multiple)  # shape = [batch_size, max_atoms, 3]
        padded_holo_pocket_coordinates = collate_tokens_coords(holo_pocket_coordinates, 0, left_pad=False,
                                                               pad_to_multiple=self.pad_to_multiple)  # shape = [batch_size, max_atoms, 3]

        padded_distance = data_utils.collate_tokens_2d(distance, 0, left_pad=False,
                                                       pad_to_multiple=self.pad_to_multiple)  # shape = [batch_size, max_atoms, max_atoms]
        padded_holo_distance = collate_cross_2d(holo_distance, 0, left_pad=False,
                                                pad_to_multiple=self.pad_to_multiple)  # shape = [batch_size, max_atoms, max_atoms]
        return padded_holo_coordinates, padded_holo_pocket_coordinates, padded_distance, padded_holo_distance, holo_center


if __name__ == '__main__':
    from unicore.data import Dictionary
    from torch.utils.data import DataLoader

    # split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
    path = './benchmark/test/data_latest.lmdb'
    pkt_dictionary = Dictionary.load('./data_provider/pkt_dict.txt')
    mol_dictionary = Dictionary.load('./data_provider/mol_dict.txt')
    pkt_dictionary.add_symbol("[MASK]", is_special=True)
    mol_dictionary.add_symbol("[MASK]", is_special=True)
    dataset = PMDataset_cid(path, pkt_dictionary, mol_dictionary, 256)
    import IPython

    IPython.embed()
    pass
