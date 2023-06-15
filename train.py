from model import BERTClass
from transformers import *
from transformers.utils import logging
# from utils import earlystopping, postdataset, preprocess
from utils import *
import torch
from torch import cuda
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
import os
import copy
from tqdm import tqdm
writer = SummaryWriter()

logging.set_verbosity_error()

path_cpt_folder = './checkpoints/'


class Train():
    def __init__(self, model, tokenizer, device, 
                fea_config, max_len=256, k_folds=5, train_ratio = 1, p_data='./dataset/all_annotated_post_784.npy'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.fea_config = fea_config
        self.optimizer = torch.optim.Adam(params = self.model.parameters()) #, lr=lr
        self.k_folds = k_folds
        self.train_ratio = train_ratio
        self.kfold_splitter= KFoldSplit(p_data, k_folds)
        self.next_folds()
        return
    
    def next_folds(self):
        # train_ratio 1 -> test set for early-stopping
        x_train, x_val, x_test = self.kfold_splitter.get_split(train_ratio = self.train_ratio)
        dataprocessor = DataPreprocessor(x_train, x_val, x_test, self.tokenizer, self.max_len, self.fea_config, self.device)
        self.train_loader = dataprocessor.get_train_dataloader()
        self.val_loader = dataprocessor.get_val_dataloader()
        self.test_loader = dataprocessor.get_test_dataloader()

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        cnt = 0
#     progress_bar = tqdm(range(len(training_loader)))
        # for p in self.model.l1.parameters():
        #     p.requires_grad = False
        for _, data in enumerate(self.train_loader, 0):
            cnt = cnt + len(data['targets'])
            # print(cnt)
            targets = data['targets'].to(self.device, dtype = torch.float)
            for k,v in data.items():
                data[k] = v.to(self.device, dtype = torch.long)
            outputs = self.model(**data)
            loss = self.model.loss_fn(outputs, targets)
            total_loss = total_loss + loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # progress_bar.update(1)
        print(f'Epoch: {epoch}, Loss:  {total_loss/cnt}')
        return total_loss/cnt

    def requires_grad_setting(self, stage = 1, verbose = True):
        # enable grad for all
        for p in self.model.parameters():
            p.requires_grad = True
            
        if stage == 1:
            for p in self.model.l1.parameters():
                p.requires_grad = False
        elif stage == 2 or stage == 12:
            for k,v in self.model.l1.named_parameters():
            #  or ('encoder.layer.9' in k) or ('encoder.layer.10' in k) or ('encoder.layer.11' in k) or ('pooler' in k)
                if ('pooler' in k): #('encoder.layer.11' in k) or 
                    v.requires_grad = True
                else:
                    v.requires_grad = False
                    
            # fix FC layers when doing stage 2 training
            # for p in self.model.l3.parameters():
            #     p.requires_grad = False
            # for p in self.model.l4.parameters():
            #     p.requires_grad = False
            
            
        if verbose == True:
            for k,v in self.model.named_parameters():
                print('{}: {}'.format(k, v.requires_grad))
        

    def training_stage(self, stage, n_epochs=1000, lr_bert = 1e-03, lr_fc = 1e-05, patience = 20, n_fold=-1):
        # self.optimizer.lr = lr
        train_loss = []
        valid_loss = []
        f1_scores = []
        if stage == 1:
            save_path = path_cpt_folder + 'checkpoint_fc'+ ('_fold_' +str(n_fold) if n_fold!=-1 else '') + '.pt'
            early_stopping = EarlyStopping(patience=patience, verbose=True, path_fc=save_path)
        elif stage == 2:
            save_path = path_cpt_folder + 'checkpoint'+ ('_fold_' +str(n_fold) if n_fold!=-1 else '') + '.pt'
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
        elif stage == 12:
            save_path = path_cpt_folder + 'checkpoint_12_'+ ('_fold_' +str(n_fold) if n_fold!=-1 else '') + '.pt'
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
            
        if stage == 1:
            self.optimizer = torch.optim.Adam([
                    {'params': t.model.l3.parameters(), 'lr': lr_fc},
                    {'params': t.model.l4.parameters(), 'lr': lr_fc}
            ])
        if stage == 2:
            self.optimizer = torch.optim.Adam([
                    {'params': t.model.l1.parameters(),'lr': lr_bert},
                    {'params': t.model.l3.parameters(), 'lr': 1e-08},
                    {'params': t.model.l4.parameters(), 'lr': 1e-08}
            ])
            
        if stage == 12:
            self.optimizer = torch.optim.Adam([
                    {'params': t.model.l1.parameters(),'lr': lr_bert},
                    {'params': t.model.l3.parameters(), 'lr': lr_fc},
                    {'params': t.model.l4.parameters(), 'lr': lr_fc}
            ])
            
        self.requires_grad_setting(stage = stage)

        for epoch in tqdm(range(n_epochs)):
            loss_train = self.train(epoch)
            # if epoch % 10 == 0:
            loss_val = self.model.validation(epoch, self.val_loader)
            # loss_tra = self.model.validation(epoch, self.train_loader)
            # print(loss_val)
#             train_loss.append(loss_train)
            # valid_loss.append(loss_eval)
#             f1_scores.append(f1_score)
            # writer.flush()
            writer.add_scalars("Loss/Fold_" + str(n_fold)+'_stage_'+str(stage), {'train': loss_train}, epoch)
            writer.add_scalars("Loss/Fold_" + str(n_fold)+'_stage_'+str(stage), {'val': loss_val}, epoch)
            # writer.add_scalars("Loss/Fold_" + str(n_fold)+'_'+str(stage), {'tra_eval': loss_tra}, epoch)
            writer.flush()
        
            if epoch%10==0:
                report = self.model.evaluation(epoch, self.test_loader)
                writer.add_scalars("Loss/Fold_" + str(n_fold)+'_stage_'+str(stage), {'test': report['loss']}, epoch)
                writer.add_scalars("F1/Fold_" + str(n_fold)+'_stage_'+str(stage), {'test': report['f1']}, epoch)
                writer.flush()
                # print(report)
                
            early_stopping(loss_val, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            if epoch == n_epochs-1:
                early_stopping.save_checkpoint(-1, self.model) 

        self.model.load_state_dict(torch.load(save_path), strict=False)
        return train_loss, valid_loss, f1_scores
    
    def evaluation(self):
        ret = self.model.evaluation(0, self.test_loader)
        # print(ret)
        return ret
        
    def evaluation_folds(self):
        pre = 0
        rec = 0
        f1 = 0
        at_1 = 0
        for f in range(0, self.k_folds):

            # loss_val = self.model.validation(0, self.train_loader)
            # print('loss_val:', loss_val)
            # if f != self.k_folds-1:
            #     self.next_folds()
                
            f_res = self.evaluation()
            if f != self.k_folds-1:
                self.next_folds()

            pre = pre + f_res['precision']
            rec = rec + f_res['recall']
            f1 = f1 + f_res['f1']
            at_1 = at_1 + f_res['at_1']
            print("loss:", f_res['loss'], "f1:", f_res['f1'])
        print('Average: precision: %3f, recall: %3f, f1: %3f, at_1: %f.' % (pre/self.k_folds, rec/self.k_folds, f1/self.k_folds, at_1/self.k_folds))

    def kfold_training(self, stage, n_epochs=1000, lr = 1e-03, patience = 20):
        f1_folds = []
        at1_folds = []
        init_params = copy.deepcopy(self.model.state_dict())
        init_opt = copy.deepcopy(self.optimizer.state_dict())
        for f in range(0, self.k_folds):
            # load initial model params
            self.model.load_state_dict(init_params, strict=True)
            self.optimizer.load_state_dict(init_opt)
            self.training_stage(stage, n_epochs, lr, patience, f)
            
            # evaluation
            result = self.model.evaluation(0, self.test_loader)
            f1_folds.append(result['f1'])
            at1_folds.append(result['at_1'])
            
            if f != self.k_folds-1:
                self.next_folds()
                
            writer.add_scalars("Folds/Micro", {'F1':result['f1'],
                                               'Precision': result['precision'],
                                               'Recall': result['recall']}, f)

            writer.add_scalar("Folds/At least One", result['at_1'], f)
            writer.flush()
                
        # print all folds performance
        print('F1-score:', f1_folds, '\nAt least one:', at1_folds)
        
    def kfold_training_two_stage(self, n_epochs=1000, lr_1 = 1e-03, lr_2 = 1e-05, patience = 20):
        # f1_folds = []
        # at1_folds = []
        init_params = copy.deepcopy(self.model.state_dict())
        init_opt = copy.deepcopy(self.optimizer.state_dict())
        for f in range(0, self.k_folds):
            # load initial model params
            self.model.load_state_dict(init_params, strict=True)
            self.optimizer.load_state_dict(init_opt)
            self.training_stage(1, n_epochs, lr_1, lr_1, patience, f)
            
            # evaluation
            result = self.model.evaluation(0, self.test_loader)
            
            writer.add_scalars("S1Folds/Micro", {'F1':result['f1'],
                                               'Precision': result['precision'],
                                               'Recall': result['recall']}, f)

            writer.add_scalar("S1Folds/At least One", result['at_1'], f)
            writer.flush()
            
            self.optimizer.load_state_dict(init_opt)
            self.training_stage(2, n_epochs, lr_2, lr_2, patience, f)
            
            result = self.model.evaluation(0, self.test_loader)
            # f1_folds.append(result['f1'])
            # at1_folds.append(result['at_1'])
            
            writer.add_scalars("S2Folds/Micro", {'F1':result['f1'],
                                               'Precision': result['precision'],
                                               'Recall': result['recall']}, f)

            writer.add_scalar("S2Folds/At least One", result['at_1'], f)
            writer.flush()
            
            
            if f != self.k_folds-1:
                self.next_folds()
                
    def kfold_training_12(self, n_epochs=1000, lr_bert= 1e-03, lr_fc = 1e-05, patience = 20):
        # f1_folds = []
        # at1_folds = []
        self.optimizer = torch.optim.Adam([
                    {'params': self.model.l1.parameters(),'lr': lr_bert},
                    {'params': self.model.l3.parameters(), 'lr': lr_fc},
                    {'params': self.model.l4.parameters(), 'lr': lr_fc}
        ])
        init_params = copy.deepcopy(self.model.state_dict())
        init_opt = copy.deepcopy(self.optimizer.state_dict())
        
        for f in range(0, self.k_folds):
            # load initial model params
            self.model.load_state_dict(init_params, strict=True)
            self.optimizer.load_state_dict(init_opt)
            self.training_stage(12, n_epochs, lr_bert, lr_fc, patience, f)
            
            # evaluation
            path_to_results = './output/'
            pt_pre = path_to_results + str(f) + '_' + 'pred.csv'
            pt_gt = path_to_results + str(f) + '_' + 'gt.csv'
            
            result = self.model.save_eval_output(pt_pre, pt_gt, self.test_loader)
            
            writer.add_scalars("S1Folds/Micro", {'F1':result['f1'],
                                               'Precision': result['precision'],
                                               'Recall': result['recall']}, f)

            writer.add_scalar("S1Folds/At least One", result['at_1'], f)
            writer.flush()
            
            if f != self.k_folds-1:
                self.next_folds()

        # print all folds performance
        # print('F1-score:', f1_folds, '\nAt least one:', at1_folds)
        
    def training_12(self, n_epochs=1000, lr_bert= 1e-03, lr_fc = 1e-05, patience = 200):
        self.optimizer = torch.optim.Adam([
                    {'params': self.model.l1.parameters(),'lr': lr_bert},
                    {'params': self.model.l3.parameters(), 'lr': lr_fc},
                    {'params': self.model.l4.parameters(), 'lr': lr_fc}
        ])

        self.training_stage(12, n_epochs, lr_bert, lr_fc, patience, 0)

        # evaluation
        path_to_results = './output/'
        pt_pre = path_to_results + '_' + 'pred.csv'
        pt_gt = path_to_results + '_' + 'gt.csv'

        result = self.model.save_eval_output(pt_pre, pt_gt, self.val_loader)

        print("Micro score on evaluation:", {'F1':result['f1'],'Precision': result['precision'],'Recall': result['recall'], 'at least one':result['at_1']})


    def kfold_training_fc(self, n_epochs=1000, lr_fc = 1e-05, patience = 200):
        # f1_folds = []
        # at1_folds = []
        self.optimizer = torch.optim.Adam([
                    {'params': self.model.l3.parameters(), 'lr': lr_fc},
                    {'params': self.model.l4.parameters(), 'lr': lr_fc}
        ])
        init_params = copy.deepcopy(self.model.state_dict())
        init_opt = copy.deepcopy(self.optimizer.state_dict())
        
        for f in range(0, self.k_folds):
            # load initial model params
            self.model.load_state_dict(init_params, strict=True)
            self.optimizer.load_state_dict(init_opt)
            self.training_stage(1, n_epochs, lr_fc=lr_fc, patience=patience, n_fold=f)
            
            # evaluation
            path_to_results = './output/'
            pt_pre = path_to_results + str(f) + '_' + 'pred.csv'
            pt_gt = path_to_results + str(f) + '_' + 'gt.csv'
            
            result = self.model.save_eval_output(pt_pre, pt_gt, self.test_loader)
            
            writer.add_scalars("S1Folds/Micro", {'F1':result['f1'],
                                               'Precision': result['precision'],
                                               'Recall': result['recall']}, f)

            writer.add_scalar("S1Folds/At least One", result['at_1'], f)
            writer.flush()
            
            if f != self.k_folds-1:
                self.next_folds()

    # def test(self):
    #     init_params = copy.deepcopy(self.model.state_dict()) # deep copy test
    #     print(self.model.evaluation(0, self.test_loader))
    #     for i in range(0,5):
    #         if i ==1:
    #             self.model.load_checkpoint(path_cpt_folder +'best/'+ 'checkpoint_fold_0.679.pt')
    #             print(self.model.evaluation(0, self.test_loader))
    #         self.model.load_state_dict(init_params, strict=True)
    #         print(self.model.evaluation(0, self.test_loader))
        

if __name__ == "__main__":
    
    os.system("rm -rf ./output/*")
    os.system("rm -rf ./runs/*")
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    fea_config = {
        'code_fea' : False, 
        'word_cnt' : True, 
        'readability' : True, 
        'sentiment' : True
    }
    
    
    loss_fn = torch.nn.BCEWithLogitsLoss()

    
    # tokenizer = AutoTokenizer.from_pretrained("jeniya/BERTOverflow")
    # model = AutoModel.from_pretrained("jeniya/BERTOverflow")

    # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    # model = AutoModel.from_pretrained("microsoft/codebert-base", output_hidden_states=True)

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # model = AutoModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
  
    # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    # model = AlbertModel.from_pretrained("albert-base-v2", output_hidden_states=True)
    
    model = BERTClass(fea_config, loss_fn, ptm = "bert-base-uncased") #'bert-base-uncased'
    model.to(device)
    
    # load pre-trained FC layer

    # tokenizer = AutoTokenizer.from_pretrained("jeniya/BERTOverflow")
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    

    
    t = Train(model, tokenizer, device, fea_config, k_folds=5, train_ratio =7/8) # when train_ratio ==1 -> testset = validation set
    
    t.kfold_training_12(n_epochs=1000, lr_bert= 1e-03, lr_fc = 1e-05, patience = 200)
    
    