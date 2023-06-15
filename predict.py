from model import BERTClass
import numpy as np
from transformers import *
from transformers.utils import logging
from utils import *
import torch
from torch import cuda
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import os, sys


def get_prediction_from_input(model, cb_clf, post):
    title = post['title']
    desc = post['description']
    code = post['code']
    if code != '':
        code_fea = cb_clf.classify(code)
    else:
        code_fea = np.zeros(5)
    

    sample = {
        'label': np.array([]),
        'title': title,
        'description': desc,
        'code_fea': code_fea
    }
    
    
    dataset = PostDataset([sample], tokenizer, max_len, fea_config, device)

    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)  
    prob, pred, gt = model.get_prediction(dl)
    print(['Discrepancy', 'Errors', 'Review', 'Conceptual', 'Learning', 'How-to', 'Other'])

    print('Prob:\t\t', np.around(prob,3))
    print('Pred:\t\t', pred)

    

if __name__ == "__main__":
    # loading checkpoint of model
    path_pre_trained = './checkpoints/wed/checkpoint_12__fold_0.pt'
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    logging.set_verbosity_error()
    
    fea_config = {
        'code_fea' : True, 
        'word_cnt' : True, 
        'readability' : True, 
        'sentiment' : True
    }
    loss_fn = torch.nn.BCEWithLogitsLoss()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len=256
    model = BERTClass(fea_config, loss_fn)
    model.to(device)
    model.load_checkpoint(path_pre_trained)
    
    cb_clf = CodeBlockClassifier(path_dataset = './dataset/codeblock/')
    
    
    post={
        'id': 'https://serverfault.com/questions/883658',
        'title': 'Failed to find `/usr/bin/docker-quickstart` when running CDH5 Docker image',
        'description': "I've downloaded the CDH 5.12 Quickstart Docker image from Cloudera but it fails to run. On the other hand, doing docker pull cloudera/quickstart:latest gives me an image for which the above works - it's just an older one (5.07 I believe). This blog post suggests that something changed around CDH 5.10. How then am I supposed to run newer images?",
         'code': '$ docker import cloudera-quickstart-vm-5.12.0-0-beta-docker.tar.gz\nsha256:8fe04d8a55477d648e9e28d1517a21e22584fd912d06de84a912a6e2533a256c\n$ docker run --hostname=quickstart.cloudera --privileged=true -t -i 8fe04d8a5547 /usr/bin/docker-quickstart\ndocker: Error response from daemon: oci runtime error: container_linux.go:265: starting container process caused "exec: \\"/usr/bin/docker-quickstart\\": stat /usr/bin/docker-quickstart: no such file or directory".\n'
    }
    

    get_prediction_from_input(model, cb_clf, post)
        # sw = input('Continue?')
        # sys.stdin.flush()
        
    
    # p_data = './dataset/all_annotated_post_784.npy'
    # posts = np.load(p_data, allow_pickle=True)
    # sample = posts[1]
    # print(sample)

    
    
    


