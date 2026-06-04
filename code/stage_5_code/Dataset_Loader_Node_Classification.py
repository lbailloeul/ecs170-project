from code.base_class.dataset import dataset
import numpy as np
import scipy.sparse as sp
import torch
import warnings


class Dataset_Loader(dataset):
    """Loader for the stage 5 citation-network node classification datasets.

    The default preprocessing and split sizes follow the GCN paper setup:
    row-normalized features, symmetric normalized adjacency with self loops,
    20 labeled training nodes per class, 500 validation nodes, and 1000 test
    nodes. The exact sampled nodes are controlled by ``seed``.
    """

    data = None

    def __init__(
        self,
        seed=0,
        dName=None,
        dDescription=None,
        labels_per_class=20,
        val_size=500,
        test_size=1000,
        split_strategy='random',
    ):
        super(Dataset_Loader, self).__init__(dName, dDescription)
        self.seed = 0 if seed is None else seed
        self.labels_per_class = labels_per_class
        self.val_size = val_size
        self.test_size = test_size
        self.split_strategy = split_strategy

    def adj_normalize(self, mx):
        """Symmetrically normalize a sparse adjacency matrix."""
        rowsum = np.array(mx.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        mx = d_mat_inv_sqrt.dot(mx).dot(d_mat_inv_sqrt)
        return mx

    def feature_normalize(self, mx):
        """Row-normalize a sparse feature matrix."""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a coalesced torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Sparse invariant checks are implicitly disabled.*',
                category=UserWarning,
            )
            try:
                return torch.sparse_coo_tensor(
                    indices,
                    values,
                    shape,
                    dtype=torch.float32,
                    check_invariants=False,
                ).coalesce()
            except TypeError:
                return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).coalesce()

    def encode_onehot(self, labels):
        classes = [str(label) for label in sorted(set(labels))]
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, [str(label) for label in labels])), dtype=np.int32)
        return onehot_labels, classes

    def _load_edges(self, edges_unordered, idx_map):
        if edges_unordered.ndim == 1:
            edges_unordered = edges_unordered.reshape(1, -1)

        mapped_edges = []
        dropped_edges = 0
        for source_id, target_id in edges_unordered[:, :2]:
            source = idx_map.get(int(source_id))
            target = idx_map.get(int(target_id))
            if source is None or target is None:
                dropped_edges += 1
                continue
            mapped_edges.append((source, target))

        if not mapped_edges:
            raise ValueError('No valid edges were found for dataset ' + str(self.dataset_name))

        if dropped_edges:
            print('Dropped', dropped_edges, 'edges with unknown node ids.')
        return np.array(mapped_edges, dtype=np.int64)

    def _sample_random_split(self, labels):
        rng = np.random.default_rng(self.seed)
        labels_np = np.asarray(labels)
        all_indices = np.arange(labels_np.shape[0])

        idx_train = []
        for class_id in sorted(np.unique(labels_np)):
            class_indices = np.where(labels_np == class_id)[0]
            if len(class_indices) < self.labels_per_class:
                raise ValueError(
                    'Class ' + str(class_id) + ' only has ' + str(len(class_indices))
                    + ' nodes, fewer than labels_per_class=' + str(self.labels_per_class)
                )
            idx_train.extend(rng.choice(class_indices, self.labels_per_class, replace=False).tolist())

        idx_train = np.array(sorted(idx_train), dtype=np.int64)
        remaining = np.setdiff1d(all_indices, idx_train, assume_unique=False)
        rng.shuffle(remaining)

        required = self.val_size + self.test_size
        if len(remaining) < required:
            raise ValueError(
                'Dataset ' + str(self.dataset_name) + ' has only ' + str(len(remaining))
                + ' non-training nodes, but validation + test requires ' + str(required)
            )

        idx_val = np.sort(remaining[:self.val_size])
        idx_test = np.sort(remaining[self.val_size:self.val_size + self.test_size])
        return idx_train, idx_val, idx_test

    def _sample_legacy_split(self, labels):
        if self.dataset_name == 'cora':
            idx_train = range(140)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'citeseer':
            idx_train = range(120)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'pubmed':
            idx_train = range(60)
            idx_test = range(6300, 7300)
            idx_val = range(6000, 6300)
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)
        else:
            raise ValueError('No legacy split is defined for dataset ' + str(self.dataset_name))
        return np.array(list(idx_train)), np.array(list(idx_val)), np.array(list(idx_test))

    def _split_summary(self, labels, idx_train, idx_val, idx_test, label_names):
        labels_np = np.asarray(labels)
        per_class_train = {
            label_names[class_id]: int(np.sum(labels_np[idx_train] == class_id))
            for class_id in sorted(np.unique(labels_np))
        }
        return {
            'strategy': self.split_strategy,
            'seed': self.seed,
            'labels_per_class': self.labels_per_class,
            'train_size': int(len(idx_train)),
            'val_size': int(len(idx_val)),
            'test_size': int(len(idx_test)),
            'train_per_class': per_class_train,
        }

    def load(self):
        """Load a citation network dataset."""
        print('Loading {} dataset...'.format(self.dataset_name))

        idx_features_labels = np.genfromtxt(
            "{}/node".format(self.dataset_source_folder_path),
            dtype=np.dtype(str),
        )
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        features = self.feature_normalize(features)
        onehot_labels, label_names = self.encode_onehot(idx_features_labels[:, -1])

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = self._load_edges(edges_unordered, idx_map)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(onehot_labels.shape[0], onehot_labels.shape[0]),
            dtype=np.float32,
        )
        adj = (adj + adj.T).tocsr()
        adj.data = np.ones_like(adj.data)
        adj.setdiag(0)
        adj.eliminate_zeros()
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0], dtype=np.float32))

        features = torch.FloatTensor(features.toarray())
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj_tensor = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        if self.split_strategy == 'legacy':
            idx_train, idx_val, idx_test = self._sample_legacy_split(labels.numpy())
        elif self.split_strategy == 'random':
            idx_train, idx_val, idx_test = self._sample_random_split(labels.numpy())
        else:
            raise ValueError("split_strategy must be either 'random' or 'legacy'.")

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        split_summary = self._split_summary(
            labels.numpy(),
            idx_train.numpy(),
            idx_val.numpy(),
            idx_test.numpy(),
            label_names,
        )
        print('Split summary:', split_summary)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {
            'node': idx_map,
            'edge': edges,
            'X': features,
            'y': labels,
            'num_classes': len(label_names),
            'label_names': label_names,
            'split_summary': split_summary,
            'utility': {'A': adj_tensor, 'reverse_idx': reverse_idx_map},
        }
        return {'graph': graph, 'train_test_val': train_test_val}
