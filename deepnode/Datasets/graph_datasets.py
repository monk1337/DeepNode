import pickle as pkl
import scipy.sparse as sp
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import numpy as np

class Karate_club(object):
    
    @staticmethod
    def load_data(path = './deepnode/Datasets/raw_datasets/Karate_club/karate.graphml'):

        
        print(os.getcwd())
        # input as a graphml file
        graph_ = nx.read_graphml(path)
        return graph_
    
    @staticmethod
    def visualize():
        graph_d = Karate_club.load_data()
        nx.draw(graph_d, cmap=plt.get_cmap('jet'),
        node_color=np.log(list(nx.get_node_attributes(graph_d, 'membership').values())))
        return plt.show()
    
    @staticmethod
    def adj_matrix():
        graph_d = Karate_club.load_data()
        adj = nx.adj_matrix(graph_d)
        return adj
    
    @staticmethod
    def feature_matrix():
        adj_matx = Karate_club.adj_matrix()
        feat_x = np.identity(n = adj_matx.shape[0])
        return feat_x
        
    @staticmethod
    def graph_preprocessing():
        adj_matx = Karate_club.adj_matrix()
        adj_tilde = adj_matx + np.identity(n = adj_matx.shape[0], dtype=np.float32)
        d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
        d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2, dtype=np.float32)
        d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
        adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt).astype(np.float32)
        return adj_norm
    
    @staticmethod
    def labels():
        
        graph_d = Karate_club.load_data()
        memberships = [m - 1 for m in nx.get_node_attributes(graph_d, 'membership').values()]
        nb_classes = len(set(memberships))
        targets = np.array([memberships], dtype=np.int32).reshape(-1)
        one_hot_targets = np.eye(4)[targets]
        
        return {'total_classes' : nb_classes, 'targets' : targets, 'one_hot' : one_hot_targets}


class loader(object):

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
        test_idx_reorder = loader.parse_index_file("./deepnode/Datasets/raw_datasets/ind.{}.test.index".format(dataset_str))
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

        nx_graph = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(nx_graph)


        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = loader.sample_mask(idx_train, labels.shape[0])
        val_mask = loader.sample_mask(idx_val, labels.shape[0])
        test_mask = loader.sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
        #
        # print(adj.shape)
        # print(features.shape)


        ################# added section to extract train subgraph ######################
        ids = set(range(labels.shape[0]))
        train_ids = ids.difference(set(list(idx_val) + list(idx_test)))
        # train_edges = [edge for edge in nx_graph.edges() if edge[0] in train_ids and edge[1] in train_ids]
        #
        # adj_train = sparse.dok_matrix((len(ids), len(ids)))
        # for edge in train_edges:
        #     if edge[0] != edge[1]:
        #         adj_train[edge[0], edge[1]] = 1

        nx_train_graph = nx_graph.subgraph(train_ids)
        adj_train = nx.adjacency_matrix(nx_train_graph)

        features = features.todense()
        features_train = features[np.array(list((train_ids)))]

        ################################################################################


        return adj_train, adj, features_train, features, labels, idx_train, idx_val, idx_test
        # G_train, G, X_train, X, Y, idx_train, idx_val, idx_test


    @staticmethod
    def prepare_graph_data(adj):
        # adapted from preprocess_adj_bias
        num_nodes = adj.shape[0]
        adj = adj + sp.eye(num_nodes)  # self-loop
        data =  adj.tocoo().data
        adj[adj > 0.0] = 1.0
        if not sp.isspmatrix_coo(adj):
            adj = adj.tocoo()
        adj = adj.astype(np.float32)
        indices = np.vstack((adj.col, adj.row)).transpose()
        return (indices, adj.data, adj.shape), adj.row, adj.col


    @staticmethod
    def prepare_graph_data1(adj):
        # adapted from preprocess_adj_bias
        num_nodes = adj.shape[0]
        adj = adj + sp.eye(num_nodes)  # self-loop
        data =  adj.tocoo().data
        if not sp.isspmatrix_coo(adj):
            adj = adj.tocoo()
        adj = adj.astype(np.float32)
        indices = np.vstack((adj.col, adj.row)).transpose()
        return (indices, adj.data, adj.shape), adj.row, adj.col
        #return (indices, adj.data, adj.shape), adj.row, adj.col, data#adj.data

    

