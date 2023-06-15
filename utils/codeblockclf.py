import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
# from sklearn.externals import joblib
import glob,os
import numpy as np


class CodeBlockClassifier():
    def __init__(self, TFIDF = True, path_dataset = '../dataset/codeblock/'):
        self.path_dataset = path_dataset
        transformer = FunctionTransformer(self.preprocess)
        
        token_pattern = r"""([A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"'])"""
        if TFIDF:
            vectorizer = TfidfVectorizer(token_pattern = token_pattern, max_features = 3000)
        else:
            vectorizer = CountVectorizer(token_pattern = token_pattern, max_features = 3000)
            
        self.prep_pipeline = Pipeline([('transformer', transformer), ('vectorizer', vectorizer)])
        
        X,y = self.datasetLoader()
        X_t= self.prep_pipeline.fit_transform(X)
        sm = SMOTE(k_neighbors=5)
        X_t_res, y_res = sm.fit_resample(X_t, y)
        
        self.clf = MultinomialNB().fit(X_t_res, y_res)

        
    def preprocess(self, x):
        return pd.Series(x).replace(r'\b([A-Za-z])\1+\b', '', regex=True)\
            .replace(r'\b[A-Za-z]\b', '', regex=True)

    
    def datasetLoader(self):
        print('Loading the dataset')
        X=[]
        y=[]
        name_file = ['code', 'markup', 'shell', 'stacktrace', 'nl']
    #     name_file= ['c', 'c#', 'c++','java', 'css', 'haskell', 'html', 'java', 'javascript', 'lua', 'objective-c', 'perl', 'php', 'python','ruby', 'r', 'scala', 'sql', 'swift', 'vb.net','markdown','bash']
        for item in name_file:
            code_loc_current=self.path_dataset+item+'/'
            file_list = glob.glob(os.path.join(code_loc_current, "*.txt"))
            for file_path in file_list:
                f=open(file_path,'r')
                data=f.read()
                label=item
                num_lines = sum(1 for line in open(file_path))
                if (num_lines>3 or item=='nl'):
                    X.append(data)
                    y.append(label)
        return X,y
    
    def classify(self, block):
    #     block = block.splitlines()
        # if(len(block.splitlines())<=3):
            # cls, prob = self.short_snippet_classifier(block)
        # else:
        cls, prob = self.long_snippet_classifier(block)
        return prob
    
    def short_snippet_classifier(self, block):
        # regex based
    #     cls_list = ['Command_line', 'error_msg', 'log', 'function', 'variable', 'short_code_snippet', 'env']

        pattern_dict = {
            'Command_line': ["~"],
            'error_msg': ["Error", "does not exist", "N|notF|found", "E|err", "Exception"],
            'log': ['at', '[A-Za-z_]\w*\b\.[A-Za-z_]\w*\b', '\:(\s)?\d+'],
            'variable': ['[A-Za-z_]+(\s)?=(\s)?[A-Za-z_]+'],  # assignment
            'code_snippet': ['F|function', '[A-Za-z_]\w*\b\.[A-Za-z_]\w*\b'],
            'url':['(^//.|^/|^[a-zA-Z])?:?/.+(/$)?'],
            'env': ["^(?:(\d+)\.)?(?:(\d+)\.)?(\*|\d+)$", "V|version"]
        }

        predicted_cls = {}

        for cls in pattern_dict.keys():
            pattern_list = pattern_dict[cls]

            cur_score = 0
            for pattern in pattern_list:
                p = re.compile(pattern)
                m = p.findall(str(block))
                # print(len(m))
                cur_score = cur_score+len(m)
            if cur_score>0:
                # print('score:', cur_score)
                predicted_cls[cls]=cur_score
        if (predicted_cls)==0:
            return 'other'
        else:
            max_score = 0
            max_cls = 'other'
            for cls in predicted_cls.items():
                if cls[1]>max_score:
                    max_score = cls[1]
                    max_cls = cls[0]
            return max_cls, 0


    def long_snippet_classifier(self, block, threshold = 0.2):
        # multinominalNB based classifier
        block_t = self.prep_pipeline.transform(block)
        cls = self.clf.predict(block_t)
        prob = self.clf.predict_proba(block_t)
        # print(prob[0])
        # acc_cls = np.argwhere(prob[0]>threshold)
        # cls = self.clf.classes_[acc_cls]
        # print(cls)
        # if(len(cls)>1):
        #     return ['mix']
        return cls, prob[0]



    # def post_contents_classifier(self, blocks):
    #     cls_list = []
    #     for block in blocks:
    #         cls = content_classifier(block)
    #         cls_list.append(cls)
    #     return cls_list



if __name__ == "__main__":
    clf = CodeBlockClassifier()
    short_block = '''
    # Get text lines
    # Consider the pdf attached to this thread. In the last page there's a row that says "Tabla 1. Indicadores.
    # Iterate util I get LTChar objects

    rows = sorted_rows
    for row in rows:          
      text_line = "".join([c.get_text() for c in row])
      print(text_line)
    '''

    print(clf.classify(short_block))
