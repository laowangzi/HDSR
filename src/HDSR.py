import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import time
import math
from tqdm import tqdm

import torch_geometric.transforms as T

from torch import Tensor
from torch_sparse import SparseTensor, matmul
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils import dropout_adj, dropout_edge

from torch_scatter import scatter_softmax

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
    
class inter_denoise_gate(torch.nn.Module):
    def __init__(self, config):
        super(inter_denoise_gate, self).__init__()
        self.W_cf = nn.Linear(config['dim'], config['dim'])
        self.W_soc = nn.Linear(config['dim'], config['dim'])
        self.W_mix = nn.Linear(config['dim'], config['dim'])
        
        self.W_en = nn.Linear(int(2*config['dim']), config['dim'])
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, embs_cf, embs_soc):
        embs = self.W_cf(embs_cf)+self.W_soc(embs_soc)+self.W_mix(embs_soc*embs_cf)
        forget = self.sig(embs)
        
        embs = self.W_en(torch.cat((embs_cf, embs_soc), -1))
        enhance = self.sig(embs)
        out = forget * embs_soc + enhance*self.tanh(self.W_soc(embs_soc))
        return out

class HDSR(torch.nn.Module):
    def __init__(self, config):
        super(HDSR, self).__init__()
        self.config = config
        self.users = config['users']
        self.items = config['items']
        self.layer = config['layer']
        self.layer_sg = config['layer_sg']
        self.dropout = config['dropout']
        self.emb_dim = config['dim']
        self.weight_decay = config['l2_reg']
        self.gcl_temp = config['gcl_temp']
        
        self.cf_weight = config['cf_weight']
        self.sg_weight = config['sg_weight']
        self.inter_gcl_weight = config['gcl_inter_weight']
        
        self.user_item_emb = nn.Embedding(self.users+self.items, self.emb_dim)
        nn.init.xavier_uniform_(self.user_item_emb.weight)
        
        self.user_social_emb = nn.Embedding(self.users+1, self.emb_dim)
        nn.init.xavier_uniform_(self.user_social_emb.weight)
        
        self.conv = torch_geometric.nn.LGConv() 
    
        self.gate = inter_denoise_gate(config)
        self.p = []
        
        self.bpr = BPRLoss()
        self.f = nn.Sigmoid()
        
        self.embs_soc = None
        self.embs_cf = None
        
    #dropout edge according to edge_weight
    def _dropout_edge(self, graph, training):
        if not training:
            edge_mask = graph.edge_index.new_ones(graph.edge_index.size(1), dtype=torch.bool)
            return graph.edge_index, edge_mask        
        row, col = graph.edge_index
        edge_weight = graph.edge_attr
        edge_mask = torch.rand(row.size(0), device=graph.edge_index.device) <= edge_weight
        edge_index = graph.edge_index[:, edge_mask]
        return edge_index, edge_mask
    
    def _cl_loss(self, emb_1, emb_2, all_layer_2):       
        pos = torch.sum(emb_1*emb_2, dim=-1) 
        tot = torch.matmul(emb_1, torch.transpose(emb_2, 0, 1)) 
        gcl_logits = tot - pos[:, None]                            
        #InfoNCE Loss
        clogits = torch.logsumexp(gcl_logits / self.gcl_temp, dim=1)
        infonce_loss = torch.mean(clogits)
        return infonce_loss
    
    def l2_reg_loss(self, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)/emb.shape[0]
        return emb_loss
        
    def compute(self, graph, emb, edge_index, layer, dtype='rec', perturbed=False): 
        all_layer = [emb]
        edge_weight = None
        if perturbed and dtype == 'rec':
            edge_index, _ = dropout_edge(edge_index, p=self.dropout, training=self.training)
        for layer in range(layer):
            if dtype == 'rec':                     
                emb = self.conv(x=emb, edge_index=edge_index)
            elif dtype == 'soc':
                if perturbed:
                    edge_index, edge_mask = self._dropout_edge(graph, training=self.training)
                    self.p.append(round(len(edge_index[0])/len(graph.edge_index[0]), 2))
                    edge_weight = graph.edge_attr[edge_mask]
                    edge_weight = scatter_softmax(edge_weight, edge_index[1], dim=-1)
                emb = self.conv(x=emb, edge_index=edge_index, edge_weight=edge_weight)
            all_layer.append(emb)
        all_layer = torch.stack(all_layer, dim=1)
        all_layer = torch.mean(all_layer, dim=1)
        return all_layer                                 
    
    def train_inter_gcl(self, graph_soc, graph_rec, user_idx): 
        all_layer_1 = self.compute(graph_soc, self.user_social_emb.weight, graph_soc.edge_index, self.layer_sg, 'soc', True) 
        all_layer_2 = self.compute(graph_rec, self.user_item_emb.weight, graph_rec.edge_index, self.layer, 'rec', True)

        all_layer_1 = F.normalize(all_layer_1, dim=1)
        all_layer_2 = F.normalize(all_layer_2, dim=1)
        
        user_emb_1 = all_layer_1[user_idx]
        user_emb_2 = all_layer_2[user_idx]

#       inter-domain denoising
        user_emb_1 = self.gate(user_emb_2, user_emb_1)
        user_emb_1 = F.normalize(user_emb_1, dim=1)           

        infonce_loss = self._cl_loss(user_emb_1, user_emb_2, all_layer_2)
        return self.inter_gcl_weight*infonce_loss
    
    def train_graph(self, graph, emb, layer, user_idx, pos_item, neg_item, dtype='rec', perturbed=False):
        all_layer = self.compute(graph, emb, graph.edge_index, layer, dtype, perturbed)       
        
        user_emb = all_layer[user_idx]
        pos_emb = all_layer[pos_item]
        neg_emb = all_layer[neg_item]
        
        pos_score = (user_emb * pos_emb).squeeze()
        neg_score = (user_emb * neg_emb).squeeze()
        bpr_loss = self.bpr(torch.sum(pos_score, dim=1), torch.sum(neg_score, dim=1))

        return bpr_loss, all_layer

    def train_sg(self, graph_soc, user_sg, pos_sg, neg_sg): 
        sg_loss, embs_soc = self.train_graph(graph_soc, self.user_social_emb.weight, self.layer_sg, user_sg, pos_sg, neg_sg, 'soc', True)
        
        users_emb_ego = self.user_social_emb(user_sg)
        pos_emb_ego = self.user_social_emb(pos_sg)
        neg_emb_ego = self.user_social_emb(neg_sg)
        reg_loss_sg = self.l2_reg_loss(users_emb_ego, pos_emb_ego, neg_emb_ego)

        sg_loss = self.sg_weight * sg_loss + reg_loss_sg * self.weight_decay

        return sg_loss
        
    def train_cf(self, graph_rec, user_idx, pos_item, neg_item): 
        cf_loss, _ = self.train_graph(graph_rec, self.user_item_emb.weight, self.layer, user_idx, pos_item, neg_item, 'rec')
        
        users_emb_ego = self.user_item_emb(user_idx)
        pos_emb_ego = self.user_item_emb(pos_item)
        neg_emb_ego = self.user_item_emb(neg_item)        
        reg_loss = self.l2_reg_loss(users_emb_ego, pos_emb_ego, neg_emb_ego)
        
        loss = self.cf_weight*cf_loss + reg_loss*self.weight_decay

        return loss
    
    def get_score_matrix(self, graph_rec):       
        all_layer = self.compute(graph_rec, self.user_item_emb.weight, graph_rec.edge_index, self.layer, 'rec', False) 

        U_e = all_layer[:self.users].detach().cpu()  # (users, concat_dim)
        V_e = all_layer[self.users:].detach().cpu()  # (items, concat_dim)
        score_matrix = torch.matmul(U_e, V_e.t())
        return score_matrix