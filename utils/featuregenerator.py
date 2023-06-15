import readability
import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer


class postFeatureGenerator():
    def __init__(self, dataset):
        self.dataset = dataset
    
    def word_cnt(self):
        for post in self.dataset:
            desc = post['description']
    #         wc = len(re.findall("[a-zA-Z_]+", desc))
            wc = len(post['description'].split())
            post['word_cnt'] = np.array([wc])
        
    def readability(self):
        for post in self.dataset:
            if len(post['description'].split())<100:
                post['readability'] = np.zeros((3))
            else:
                results = readability.getmeasures(post['description'])
                f_flesch_kincaid = results['readability grades']['FleschReadingEase'] # r.flesch_kincaid()
                f_smog = results['readability grades']['SMOGIndex'] #r.smog(all_sentences=True)
                f_ari = results['readability grades']['ARI']  #r.ari()
                post['readability'] = np.array([f_flesch_kincaid, f_smog, f_ari])

    def sentiment(self):
        sia = SentimentIntensityAnalyzer()
        for post in self.dataset:
            res = sia.polarity_scores(post['description'])
            post['sentiment'] = np.array([res['neg'], res['neu'], res['pos'], res['compound']])
                