from transformers import *
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split

from .featuregenerator import postFeatureGenerator


class PostDataset(Dataset):

    def __init__(self, data, tokenizer, max_len, fea_config, device):
        self.tokenizer = tokenizer
        self.posts = data
#         self.targets = label
        self.max_len = max_len
        self.fea_config = fea_config
        self.device = device
        self.cls = ['Discrepancy', 'Errors', 'Review', 'Conceptual', 'Learning', 'How-to', 'Other']
        self.fea_init()

    def __len__(self):
        return len(self.posts)
    
    def fea_init(self):
        fg = postFeatureGenerator(self.posts)
        for key, sw in self.fea_config.items():
            if key=='code_fea':
                pass
            else:
                eval('fg.'+key)()
        
    def fea_generator(self, post):
        fea = np.array([], dtype= np.float)
        # print(post.keys())
        for key, sw in self.fea_config.items():
            if sw == True:
                fea = np.append(fea, post[key])
        # return torch.tensor(fea, dtype=torch.long)
        fea = torch.from_numpy(fea)
        return fea.float()
    
    def multi_label_transfrom(self, p):
        label = np.zeros((len(self.cls)))
        for lb in p['label']:
            label[self.cls.index(lb)] = 1
        return torch.tensor(label, dtype=torch.float)
        
    def __getitem__(self, index):
        post = self.posts[index]
#         print(post)
        title_text = post['title']
        desc_text = post['description']

        t_inputs = self.tokenizer.encode_plus(
            title_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation = 'longest_first',
            return_token_type_ids=True
        )
        t_ids = t_inputs['input_ids']
        t_mask = t_inputs['attention_mask']
        t_token_type_ids = t_inputs["token_type_ids"]
        
        
        d_inputs = self.tokenizer.encode_plus(
            desc_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation = 'longest_first',
            return_token_type_ids=True
        )
        d_ids = d_inputs['input_ids']
        d_mask = d_inputs['attention_mask']
        d_token_type_ids = d_inputs["token_type_ids"]

        
        fea = self.fea_generator(post)
        label = self.multi_label_transfrom(post)
        

        return {
            't_ids': torch.tensor(t_ids, dtype=torch.long).to(self.device),
            't_mask': torch.tensor(t_mask, dtype=torch.long).to(self.device),
            't_token_type_ids': torch.tensor(t_token_type_ids, dtype=torch.long).to(self.device),
            
            'd_ids': torch.tensor(d_ids, dtype=torch.long).to(self.device),
            'd_mask': torch.tensor(d_mask, dtype=torch.long).to(self.device),
            'd_token_type_ids': torch.tensor(d_token_type_ids, dtype=torch.long).to(self.device),
            
            'features': fea.clone().detach().to(self.device),
            'targets': label.to(self.device)

        }
    
    
