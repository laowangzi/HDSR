from torch_scatter import scatter_softmax, scatter_min, scatter_max
import torch
import seaborn as sns
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm

def compute_edge_weight(loader, social_index, config):
    feature_range = config['feature_range']
    trustee = social_index[0]
    trustor = social_index[1]

    G = nx.DiGraph()
    nodes = list(set(trustee) | set(trustor))
    edges = [(trustor[idx], trustee[idx]) for idx in range(len(trustor))]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    if config['edge_weight'] == 'influence':
        edge_weight = cal_influence(loader, social_index)
    else:
        edge_weight = cal_global(G, social_index, config)  

    edge_weight = scatter_minmax(torch.FloatTensor(edge_weight), torch.LongTensor(social_index[1]), feature_range)
    
    return edge_weight

def scatter_minmax(data, index, feature_range):
    min_values = scatter_min(data, index)[0]
    max_values = scatter_max(data, index)[0]
    #avoid divide 0, +1e-5
    scaled_data = (data - torch.gather(min_values, dim=0, index=index)) / (torch.gather(max_values, dim=0, index=index) - torch.gather(min_values, dim=0, index=index) + 1e-5)
    scaled_data = scaled_data*(feature_range[1]-feature_range[0]) + feature_range[0]
    return scaled_data

#u-i union num
def cal_influence(loader, social_index):
    train = loader.train
    social = loader.trust

    his_dict = {}
    user_group = train.groupby('userid')
    for u, v in user_group:
        his_dict[u] = set(v['itemid'].to_list())

    edge_weight = []
    for i in range(len(social_index[0])):
        trustee = social_index[0][i]
        trustor = social_index[1][i]
        if trustee in his_dict.keys() and trustor in his_dict.keys():
#             edge_weight.append(gamma+len(his_dict[trustor]&his_dict[trustee]))
            union = len(his_dict[trustor]&his_dict[trustee])
            edge_weight.append(union/len(his_dict[trustor]))
        else:
            edge_weight.append(0)
    return edge_weight

def cal_global(G, social_index, config):
    edge_weight = []
    dtype = config['edge_weight']                
    centrality = G.degree()
    for src in social_index[0]:
        edge_weight.append(centrality[src])
    return edge_weight
    