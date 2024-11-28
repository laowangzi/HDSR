import os 
import numpy as np
import pandas as pd

import json
import time
from torch import optim
from Data import *
from torch_geometric.data import Data
from HDSR import HDSR
# from evaluate import eval_model
from evaluate_multiproc import eval_model
from Influence import compute_edge_weight
from prettytable import PrettyTable

import warnings
warnings.filterwarnings('ignore')

Ks = [20, 50, 100]
config = {}
config['dataset'] = 'Douban'

ds = config['dataset']
config['save_path'] = f'./ckpts/log_{ds}_1.pt'
config['device'] = 'cuda:1'
config['dim'] = 64
config['num_epoch'] = 300

#load the best hyperparames from config folder
with open(f'./config/config_{ds}.json', 'r') as f:
    best_hyperparam = json.load(f)

for k in best_hyperparam.keys():
    config[k] = best_hyperparam[k]
for k, v in config.items():
    print(f'{k}: {v}')

seed = 2024
print(f'seed: {seed}')
init_seed(seed, True)

loader = MyLoader(config)
merge_df = loader.train
#recommendation domain
src = merge_df.loc[:, 'userid'].to_list()+ merge_df.loc[:, 'itemid'].to_list()
tgt = merge_df.loc[:, 'itemid'].to_list() + merge_df.loc[:, 'userid'].to_list()
edge_index = [src, tgt]

#social domain
src_social = loader.trust['trustee'].to_list()
tgt_social = loader.trust['trustor'].to_list()
social_index = [src_social, tgt_social]

config['edge_weight'] = 'degree'
edge_weight_view1 = compute_edge_weight(loader, social_index, config)
config['edge_weight'] = 'influence'
edge_weight_view2 = compute_edge_weight(loader, social_index, config)
edge_weight = (edge_weight_view1+edge_weight_view2)/2


edge_index = torch.LongTensor(edge_index)
social_index = torch.LongTensor(social_index)
graph_rec = Data(edge_index=edge_index.contiguous())
graph_soc = Data(edge_index=social_index.contiguous(), edge_attr=edge_weight)
graph_rec = graph_rec.to(config['device'])
graph_soc = graph_soc.to(config['device'])

# mix_train_loader, dataset = loader.get_mix_loader()
train_loader_rec, dataset_rec = loader.get_cf_loader(bs=config['trainbatch'])
train_loader_soc, dataset_soc = loader.get_sg_loader(bs=config['trainbatch'])

model = HDSR(config)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
model = model.to(config['device'])

best_score  = 0.0
patience = config['patience']

for epoch in range(config['num_epoch']):
    loss_list = []
    start = time.time()
    model.train()
    dataloader_iterator = iter(train_loader_soc)
    for batch_data in train_loader_rec:
        try:
            batch_data_soc = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_loader_soc)
            batch_data_soc = next(dataloader_iterator)
        user_sg, pos_sg, neg_sg = batch_data_soc
        user_sg, pos_sg, neg_sg = user_sg.to(config['device']), pos_sg.to(config['device']), neg_sg.to(config['device'])
        user, pos, neg = batch_data
        user, pos, neg = user.to(config['device']), pos.to(config['device']), neg.to(config['device'])
        optimizer.zero_grad()
        cf_loss = model.train_cf(graph_rec, user, pos, neg)
        sg_loss = model.train_sg(graph_soc, user_sg, pos_sg, neg_sg)
        inter_gcl_loss = model.train_inter_gcl(graph_soc, graph_rec, user)
        loss = cf_loss + sg_loss + inter_gcl_loss
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().cpu().numpy())
    if (epoch+1) % 1 == 0:
        result = eval_model(model, loader, 'valid', graph_rec)
        print('Valid Performance')
        for i, k in enumerate(Ks):
            table = PrettyTable([f'recall@{k}',f'ndcg@{k}',f'precision@{k}',f'hit_ratio@{k}'])
            table.add_row([round(result['recall'][i], 4),round(result['ndcg'][i], 4),round(result['precision'][i], 4), round(result['hit_ratio'][i], 4)])
            print(table)
        curr_score = result['recall'][1] + result['ndcg'][1] + result['recall'][2] + result['ndcg'][2]
        if curr_score > best_score:
            best_score = curr_score
            save(model, config)
            print('best model save at:', config['save_path'])
            patience = config['patience']
        else:
            patience -= 1
        if patience <= 0:
            break
    end = time.time()
    epoch_loss = round(np.mean(loss_list), 4)
    print(f'Epoch: {epoch+1}, epoch time: {round(end-start, 2)}s, epoch loss: {epoch_loss}')
    print('-'*90)
print(f'mean dropout rate:', round(np.mean(model.p), 2))
model = load_best(model, config)
print('load best model from:', config['save_path'])
result = eval_model(model, loader, 'test', graph_rec)
print('Test Performance:')
for i, k in enumerate(Ks):
    table = PrettyTable([f'recall@{k}',f'ndcg@{k}',f'precision@{k}',f'hit_ratio@{k}'])
    table.add_row([round(result['recall'][i], 4),round(result['ndcg'][i], 4),round(result['precision'][i], 4), round(result['hit_ratio'][i], 4)])
    print(table)
