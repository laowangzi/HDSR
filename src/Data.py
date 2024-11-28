import os
import random
import collections
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class MyLoader(object):
    def __init__(self, config):
        self.config = config
        self.loadData()
        
    def loadData(self):
        path = '../data/'+self.config['dataset']+'/'
        
        self.train = pd.read_csv(path+f'train.txt', header=None, sep='\t')
        self.train.columns = ['userid', 'itemid', 'rating']
        self.valid = pd.read_csv(path+f'valid.txt', header=None, sep='\t')
        self.valid.columns = ['userid', 'itemid', 'rating']
        self.test = pd.read_csv(path+f'test.txt', header=None, sep='\t')
        self.test.columns = ['userid', 'itemid', 'rating']
        
        self.trust = pd.read_csv(path+'SG.txt', header=None, sep='\t')
        self.trust.columns = ['trustor', 'trustee']

        trust_users = set(self.trust['trustor'].values.tolist())|set(self.trust['trustee'].values.tolist())
        
#         df = pd.concat([self.train, self.valid, self.test], 0)
        df = pd.concat([self.train, self.valid, self.test])
        self.users = len(set(df['userid'].values.tolist())|trust_users)
        self.items = len(df['itemid'].unique())
        self.config['users'] = self.users
        self.config['items'] = self.items

        self.statistic(df, self.trust)
        
        #re-hash itemid
        self.train['itemid'] = self.train['itemid'].apply(lambda x: x+self.users)
        self.valid['itemid'] = self.valid['itemid'].apply(lambda x: x+self.users)
        self.test['itemid'] = self.test['itemid'].apply(lambda x: x+self.users)
        
        self.prepare_eval()
        
    
    def statistic(self, df, trust):
        user_num = len(df['userid'].unique())
        item_num = len(df['itemid'].unique())
        sparse = len(df)/(user_num*item_num)
        interaction = len(df)
        print(f'users num={user_num}, item num={item_num}, ratings={interaction}, sparse={sparse}')
        links = len(trust)
        linkUser = len(set(trust['trustor'].values.tolist())|set(trust['trustee'].values.tolist()))
        print(f'link user={linkUser}, link num={links}')
    
    def get_cf_loader(self, bs=1024):
        dataset = TrainDataset(self.train, self.rated_dict, self.users, self.items)            
        return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=bs, pin_memory=True, drop_last=True), dataset
    
    def get_eval_data(self, dtype='valid'):
        if dtype == 'valid':
            return self.rated_dict, self.item_dict
        else:
            return self.rated_dict, self.item_dict_test
    def get_sg_loader(self, bs=1024):
        dataset = sgDataset(self.trust, self.users)
        return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=bs, drop_last=True), dataset
    
    def prepare_eval(self):
        eval_users = list(self.valid['userid'].unique())
        eval_df = self.valid

        self.item_dict = {}
        item_group = eval_df.groupby('userid')
        for u, v in item_group:
            self.item_dict[u] = v['itemid'].values.tolist()

        self.rated_dict = {}
        rated_group = self.train.groupby('userid')
        for u, v in rated_group:
            self.rated_dict[u] = set(v['itemid'].values.tolist())

        eval_users = list(self.test['userid'].unique())
        eval_df = self.test

        self.item_dict_test = {}
        item_group = eval_df.groupby('userid')
        for u, v in item_group:
            self.item_dict_test[u] = v['itemid'].values.tolist()    

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, rated_dict, user_num, item_num):
        self.pos = df
        self.pos_rec = rated_dict 
        self.user_num = user_num
        self.item_num = item_num        
        self.train_cf = df.values
    
    def generate_cf_neg(self, user):
        poss = self.pos_rec[user]
        while True:
            neg = random.randint(self.user_num, self.user_num+self.item_num-1)
            if neg not in poss:
                break
        return neg
            
    def __len__(self):
        return len(self.pos)
    
    def __getitem__(self, index):
        user = self.train_cf[index][0]
        pos = self.train_cf[index][1]
        neg = self.generate_cf_neg(user)
        return user, pos, neg
    
class sgDataset(torch.utils.data.Dataset):
    def __init__(self, trust, user_num):
        self.sg = trust 
        self.pos_soc = self.generate_soc_pos()        
        self.user_num = user_num        
        self.train_sg = trust.values

    def generate_soc_pos(self):
        soc_pos = {}
        sg_group = self.sg.groupby('trustor')
        for u, v in sg_group:
            soc_pos[u] = v['trustee'].values.tolist()
        return soc_pos
    
    def generate_sg_neg(self, trustor):
        poss = self.pos_soc[trustor]
        while True:
            neg = random.randint(0, self.user_num-1)
            if neg not in poss:
                break
        return neg    
            
    def __len__(self):
        return len(self.sg)
    
    def __getitem__(self, index):
        row = self.train_sg[index] 
        trustor = self.train_sg[index][0]
        pos_sg = self.train_sg[index][1]
        neg_sg = self.generate_sg_neg(trustor)
        return trustor, pos_sg, neg_sg
    
    
# Checkpoints
def save(model, config):
    torch.save(model.state_dict(),config['save_path'])
    save = config['save_path']
#     print(f'best model save at {save}')
def load_best(model, config):
    model.load_state_dict(torch.load(config['save_path']))
    save = config['save_path']
#     print(f'load model from {save}')
    return model
            
def init_seed(seed, reproducibility):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False