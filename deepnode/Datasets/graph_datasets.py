import pickle as pkl
import scipy.sparse as sp
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
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