import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from sklearn import metrics
import os
from .postdataset import PostDataset

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4




class KFoldSplit():
    def __init__(self, p_data, k_folds = 5):
        self.k_folds = k_folds
        self.data = np.load(p_data, allow_pickle=True)
        self.kfold = KFold(n_splits=k_folds, shuffle=False)
        self.folds = [f[1] for f in self.split()]
        self.cur_test_fold = 0
        
        # print(folds)
        
    def split(self):
        return self.kfold.split(self.data)
    
    def get_split(self, train_ratio = -1, save_test = True): # one out of four
        # train_ratio == 1: testset for early-stopping  -1: one fold for early-stopping
        i_test = self.folds[self.cur_test_fold]
        x_test = self.data[i_test]
        mask = np.ones(len(self.data), bool)
        mask[i_test] = False
        x_train_val = self.data[mask]
        self.cur_test_fold = self.cur_test_fold+1
        assert (self.cur_test_fold<=self.k_folds), 'Out of boundary.'
        
        if train_ratio == 1:
            x_train = x_train_val
            x_val= x_test
        else:
            if train_ratio == -1:
                train_ratio = (self.k_folds-2) / (self.k_folds-1)
            x_train, x_val = train_test_split(x_train_val, test_size=1-train_ratio, random_state=0)
        # print(len(x_train), len(x_val), len(x_test))
        
        if save_test == True:
            output_folder_path = './output/'
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            np.save(os.path.join(output_folder_path, str(self.cur_test_fold-1) + '_testset.npy'), x_test, allow_pickle=True)
        return x_train, x_val, x_test
    

class DataPreprocessor():
    def __init__(self, x_train, x_val, x_test, tokenizer, max_len, fea_config, device):
        self.x_train, self.x_val, self.x_test = x_train, x_val, x_test
        self.tokenizer = tokenizer
        self.training_set = PostDataset(self.x_train, tokenizer, max_len, fea_config, device)
        self.validation_set = PostDataset(self.x_val, tokenizer, max_len, fea_config, device)
        self.testing_set = PostDataset(self.x_test, tokenizer, max_len, fea_config, device)
    
    def get_train_dataloader(self):
        params = {'batch_size': TRAIN_BATCH_SIZE,
                        'shuffle': False,
                        'num_workers': 0
                        }
        return DataLoader(self.training_set, **params)
    
    def get_val_dataloader(self):
        params = {'batch_size': VALID_BATCH_SIZE,
                        'shuffle': False,
                        'num_workers': 0
                        }
        return DataLoader(self.validation_set, **params)

    def get_test_dataloader(self):
        params = {'batch_size': VALID_BATCH_SIZE,
                        'shuffle': False,
                        'num_workers': 0
                        }
        return DataLoader(self.testing_set, **params)
    
    
if __name__ == "__main__":
    print("data processor.")
    p_data = '../dataset/all_annotated_post_784.npy'
    k_fold_obj = KFoldSplit(p_data, 5)
    # folds = k_fold_obj.split()
    x_train, x_val, x_test = k_fold_obj.get_split()
    print(x_test[0])
#     from transformers import *
#     from torch import cuda
    
#     fea_config = {
#         'code_fea' : False, 
#         'word_cnt' : False, 
#         'readability' : False, 
#         'sentiment' : False
#     }
#     device = 'cuda' if cuda.is_available() else 'cpu'
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     dp = DataPreprocessor(x_train, x_val, x_test, tokenizer, 128, fea_config, device)
    
#     vdl = dp.get_val_dataloader()