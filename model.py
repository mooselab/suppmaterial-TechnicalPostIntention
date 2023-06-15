from transformers import *
import torch
import numpy as np
from sklearn.metrics import classification_report
import os
# MAX_LEN = 256
# TRAIN_BATCH_SIZE = 16
# VALID_BATCH_SIZE = 4
# EPOCHS = 500
# LEARNING_RATE = 1e-04
# LEARNING_RATE_S2 = 1e-05



class BERTClass(torch.nn.Module):
    def __init__(self, fea_config, loss_fn, ptm=None): 
        super(BERTClass, self).__init__()
        
        self.loss_fn = loss_fn
        
        fea_dim = {
            'code_fea' : 5,
            'word_cnt' : 1,
            'readability' : 3,
            'sentiment' : 4
        }
        
        self.dim_emb = 768*2
        
        # for key,val in fea_config.items():
        #     if val==True:
        #         self.dim_linear = self.dim_linear + fea_dim[key]
        
        self.dim_fea = 0
        for key,val in fea_config.items():
            if val==True:
                self.dim_fea = self.dim_fea + fea_dim[key]
        
        if ptm != None:
            self.l1 = BertModel.from_pretrained(ptm, output_hidden_states=True)
            # self.l1 = AutoModel.from_pretrained(ptm, output_hidden_states=True)
            # self.l1 = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
            # self.l1 = DistilBertForSequenceClassification.from_pretrained(ptm, output_hidden_states=True, num_labels=768)
            # self.l1 = AlbertModel.from_pretrained("albert-base-v2", output_hidden_states=True)
            
        else:
            self.l1 = BertModel(BertConfig())
#         self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(self.dim_emb, 50)
        self.l4 = torch.nn.Linear(50 + self.dim_fea, 7)
#         self.l4 = torch.nn.Linear(self.dim_linear, 7)
    
    def forward(self, t_ids, t_mask, t_token_type_ids, d_ids, d_mask, 
                d_token_type_ids, features, targets):
        output_title= self.l1(t_ids , attention_mask = t_mask , token_type_ids = t_token_type_ids)['pooler_output']
        output_desc= self.l1(d_ids, attention_mask = d_mask, token_type_ids = d_token_type_ids)['pooler_output']
        
        # output_title= self.l1(t_ids , attention_mask = t_mask)['logits']
        # output_desc= self.l1(d_ids, attention_mask = d_mask)['logits']
        
        bert_emb = torch.cat((output_title, output_desc), dim=1)
        
#         output_2 = self.l2(combined)
        output_3 = self.l3(bert_emb)
        combined = torch.cat((output_3, features), dim=1)
        output = self.l4(combined)
        return output

    def load_checkpoint(self, p_cpt):
        self.load_state_dict(torch.load(p_cpt), strict=False)
        
    
    def validation(self, epoch, dataloader):
        self.eval()
        with torch.no_grad():
    #         progress_bar = tqdm(range(len(dataloader)))
            total_loss = 0
            cnt = 0
            for _, data in enumerate(dataloader, 0):
                targets = data['targets'] #.to(self.device, dtype = torch.float)
                cnt = cnt + len(targets)
                # for k,v in data.items():
                #     data[k] = v.to(self.device, dtype = torch.long)
                outputs = self.forward(**data)
                loss = self.loss_fn(outputs, targets)
                total_loss = total_loss + loss.item()
            loss = total_loss/cnt
        return loss

    def save_eval_output(self, pt_pre, pt_gt, dataloader):
        ret = self.evaluation(0, dataloader)
        outputs = ret['prediction']
        targets = ret['groundtruth']
        # print(targets[2])
        # print(loss)
        outputs = np.array(outputs)
        path_dir = os.path.dirname(pt_pre)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir) 
        np.savetxt(pt_pre, outputs, delimiter=',', fmt='%f', header='Discrepancy, Errors, Review, Conceptual, Learning, How-to, Other')
        np.savetxt(pt_gt, targets, delimiter=',', fmt='%f', header='Discrepancy, Errors, Review, Conceptual, Learning, How-to, Other')
        return ret
    
    
    def generate_predict_label(self, pred_prob):
        ret = []
        shape = pred_prob.shape
        n_pred = len(pred_prob)
        for i in range(n_pred):
            lb = np.array(pred_prob[i]>=0.5)
            if sum(lb)==0:
                lb = np.zeros(shape[1])
                lb[np.argmax(pred_prob[i])]=1
                lb = np.array(lb, dtype=bool)
            elif lb[-1] == 1:
                lb = np.zeros(shape[1])
                lb[-1] = 1
                lb = np.array(lb, dtype=bool)
            ret.append(lb)
        return np.array(ret)
    
    def get_prediction(self, dataloader):
        self.eval()
        with torch.no_grad():
            for _, data in enumerate(dataloader, 0):
                # print(data)
                targets = data['targets']
                outputs = self.forward(**data)
                
                outputs = torch.sigmoid(outputs).cpu().detach().numpy()
                targets = targets.cpu().detach().numpy().tolist()
        target = np.array(targets[0], dtype=bool)
        # print(np.around(outputs,3)[0])
        pred_label = self.generate_predict_label(outputs)[0]
        # print(label)
        return outputs[0], pred_label, target
    
    def evaluation(self, epoch, dataloader):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            fin_targets=[]
            fin_outputs=[]
            cnt = 0
            for _, data in enumerate(dataloader, 0):
                cnt = cnt + len(data['targets'])
                targets = data['targets'] #.to(self.device, dtype = torch.float)
                # for k,v in data.items():
                #     data[k] = v.to(self.device, dtype = torch.long)
                outputs = self.forward(**data)
                loss = self.loss_fn(outputs, targets)
                total_loss = total_loss + loss.item()
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            loss = total_loss/cnt
            

        # fin_outputs_lb = np.array(fin_outputs) >= 0.5
        fin_outputs_lb = self.generate_predict_label(np.array(fin_outputs))
        
        report = classification_report(
        fin_targets,
        fin_outputs_lb,
        output_dict=True,
        target_names=['Discrepancy', 'Errors', 'Review', 'Conceptual', 'Learning', 'How-to', 'Other'],
        zero_division = 0
        )
        # print(report)
        # accuracy = metrics.accuracy_score(targets, outputs)
        # f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        # f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        cnt = 0
        for i in range(len(fin_outputs_lb)):
            if (np.logical_and(fin_outputs_lb[i], fin_targets[i])).any()==True:
                cnt=cnt+1
                

        return {
            'epoch': epoch,
            'loss': loss,
            'precision': report['micro avg']['precision'],
            'recall': report['micro avg']['recall'],
            'f1': report['micro avg']['f1-score'],
            'at_1': cnt/len(fin_outputs_lb),
            'prediction': fin_outputs,
            'groundtruth': fin_targets
        }
    
    
if __name__ == "__main__":
    
    fea_config = {
        'code_fea' : True, 
        'word_cnt' : True, 
        'readability' : True, 
        'sentiment' : True
    }

    model = BERTClass(fea_config) # append loss fn
    model.to(device)

