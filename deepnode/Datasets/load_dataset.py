import pickle as pkl
import scipy.sparse as sp
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import numpy as np

class graph_preprocessing(object):
    
    @staticmethod
    def transform_semi_supervise(labels, mask_nodes_):

        # labels : node labels as list [ 1,2,3,4,5]
        # mask_nodes : no of nodes will be use for semi-supervised traning let say we have 5 classes total 34 samples then I am using 
        # 5 labels for ssl setting so mask_nodes will be 5
    
        nb_node_classes = len(set(labels))
        
        targets = np.array([labels], dtype=np.int32).reshape(-1)
        one_hot_nodes = np.eye(nb_node_classes)[targets]
        
        
        # Pick one at random from each class
        labels_to_keep = [np.random.choice(
        np.nonzero(one_hot_nodes[:, c])[0]) for c in range(mask_nodes_)]
        
        y_train = np.zeros(shape=one_hot_nodes.shape,
                    dtype=np.float32)
        y_val = one_hot_nodes.copy()
        
        train_mask = np.zeros(shape=(len(labels),), dtype=np.bool)
        val_mask = np.ones(shape=(len(labels),), dtype=np.bool)
        
        
        for l in labels_to_keep:
            y_train[l, :] = one_hot_nodes[l, :]
            y_val[l, :] = np.zeros(shape=(nb_node_classes,))
            train_mask[l] = True
            val_mask[l] = False
            
        return {
                'all_labels': labels, 
                'train_labels': y_train, 
                'val_labels': y_val, 
                'train_mask': train_mask, 
                'val_mask': val_mask 
                }
    @staticmethod
    def sparse_to_tuple(sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    @staticmethod
    def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    @staticmethod
    def preprocess_adj(adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = graph_preprocessing.normalize_adj(adj + sp.eye(adj.shape[0]))
        return graph_preprocessing.sparse_to_tuple(adj_normalized)


    @staticmethod
    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return graph_preprocessing.sparse_to_tuple(features)

    
    

class graph_dataset(object):

    @staticmethod
    def load_random_data(size):

        adj = sp.random(size, size, density=0.002) # density similar to cora
        features = sp.random(size, 1000, density=0.015)
        int_labels = np.random.randint(7, size=(size))
        labels = np.zeros((size, 7)) # Nx7
        labels[np.arange(size), int_labels] = 1

        train_mask = np.zeros((size,)).astype(bool)
        train_mask[np.arange(size)[0:int(size/2)]] = 1

        val_mask = np.zeros((size,)).astype(bool)
        val_mask[np.arange(size)[int(size/2):]] = 1

        test_mask = np.zeros((size,)).astype(bool)
        test_mask[np.arange(size)[int(size/2):]] = 1

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
    
        # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

    @staticmethod
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index
    
    @staticmethod
    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)
    
    @staticmethod
    def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("./deepnode/Datasets/raw_datasets/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = graph_dataset.parse_index_file("./deepnode/Datasets/raw_datasets/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = graph_dataset.sample_mask(idx_train, labels.shape[0])
        val_mask = graph_dataset.sample_mask(idx_val, labels.shape[0])
        test_mask = graph_dataset.sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        print(adj.shape)
        print(features.shape)

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


    

