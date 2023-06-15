import argparse
import pathlib
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
from tabulate import tabulate
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Control the output of metrics.')
parser.add_argument('pred_path', type=pathlib.Path, help='Path to the prediction file (CSV).')
parser.add_argument('gt_path', type=pathlib.Path, help='Path to the groudtruth file (CSV).')
parser.add_argument('--post', type=pathlib.Path, help='Path to the groudtruth file (CSV).') #, default=os.devnull
parser.add_argument('-k', type=int, default=3, help='P/R/F1@K')
parser.add_argument('-o', '--overall', action='store_true', default=False, help='Switch for overall micro average metrics.') 
parser.add_argument('-a', '--atk', action='store_true', default=False, help='Switch for at K metrics.') 
parser.add_argument('-p', '--perclass', action='store_true', default=False, help='Switch for per-class metrics.') 
parser.add_argument('-e', '--error', action='store_true', default=False, help='Print the incorrect items.')
parser.add_argument('-i', type=int, default=-1, help='Print original post according to post id.')
parser.add_argument('-r', type=int, default=3, help='Decimal place of the results.')

# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
args = parser.parse_args()
print(args)


class Evaluation:
    def __init__(self, at_k_value, dec_place, pt_pred='prediction.csv', pt_gt='groundtruth.csv', pt_post='../data/all_annotated_post_784.npy'):
        self.pred_prob = np.genfromtxt(pt_pred, delimiter=',') # , names = False,dtype=np.float64
        self.ground_truth = np.genfromtxt(pt_gt, delimiter=',')
        # self.pred_lb = self.pred_prob>=0.5
        self.pred_lb = self.generate_predict_label()
        self.n_sample = len(self.ground_truth)
        self.n_cls = self.ground_truth.shape[1]
        self.at_k_value = at_k_value
        self.dec_place = dec_place
        self.target_names = ['Discrepancy', 'Errors', 'Review', 'Conceptual', 'Learning', 'How-to', 'Other']
        posts = np.load(pt_post, allow_pickle=True)
        self.x_test = posts
        # with original dataset:
        # _, self.x_test = train_test_split(posts, test_size=1-0.8, random_state=0)
        
        # self.target_names=['Conceptual', 'Discrepancy', 'Errors', 'How-to', 'Learning', 'Other', 'Review']
        
        assert len(self.pred_prob) == len(self.ground_truth), 'Length mismatch.'
        
        print(str(self.n_sample) + " testing samples.\n")

        
    def generate_predict_label(self):
        ret = []
        shape = self.pred_prob.shape
        n_pred = len(self.pred_prob)
        for i in range(n_pred):
            lb = np.array(self.pred_prob[i]>=0.5)
            if sum(lb)==0:
                lb = np.zeros(shape[1])
                lb[np.argmax(self.pred_prob[i])]=1
                lb = np.array(lb, dtype=bool)
            elif lb[-1] == 1:
                lb = np.zeros(shape[1])
                lb[-1] = 1
                lb = np.array(lb, dtype=bool)
            if lb[0]==0 and lb[1]==1:
                lb[0] = 1
            ret.append(lb)
        # print(ret)
        return np.array(ret)
        
        
    def print_overall_metrics(self):
        report = classification_report(
                self.ground_truth,
                self.pred_lb,
                output_dict=True,
                target_names=self.target_names,
                zero_division = 0
                )

        # At least one
        cnt = 0
        for i in range(len(self.pred_lb)):
            if (np.logical_and(self.pred_lb[i], self.ground_truth[i])).any()==True:
                cnt=cnt+1

        results = {
            'precision': round(report['micro avg']['precision'], self.dec_place),
            'recall': round(report['micro avg']['recall'], self.dec_place),
            'f1': round(report['micro avg']['f1-score'], self.dec_place),
            'at_least_1': round(cnt/self.n_sample, self.dec_place)
        }
        print('Overall Micro Average:')
        print(tabulate(list([results.values()]), list(results.keys())), '\n')
        return results

    
    
    def binary_report(self, i):
        class_name = self.target_names[i]
        accuracy = sum(self.ground_truth[:,i]==self.pred_lb[:,i])/self.n_sample
        # print(self.ground_truth[:,i])
        AUC = 0
        try:
            AUC = metrics.roc_auc_score(self.ground_truth[:,i], self.pred_lb[:,i])

        except ValueError:
            pass
        MCC = metrics.matthews_corrcoef(self.ground_truth[:,i], self.pred_lb[:,i])
        prec = metrics.precision_score(self.ground_truth[:,i], self.pred_lb[:,i] ,zero_division = 0)
        rec = metrics.recall_score(self.ground_truth[:,i], self.pred_lb[:,i])
        f1_score = metrics.f1_score(self.ground_truth[:,i], self.pred_lb[:,i])
        
        
        return {
            'class': class_name,
            'accuracy': round(accuracy, self.dec_place),
            'auc': round(AUC, self.dec_place),
            'mcc': round(MCC, self.dec_place),
            'precision': round(prec, self.dec_place),
            'recall': round(rec, self.dec_place),
            'f1_score': round(f1_score, self.dec_place)
        }
    
    def print_per_class_results(self):
        results_for_cls = []
        for i in range(self.n_cls):
            results_for_cls.append(self.binary_report(i))
        print('Per-class results:')
        print(tabulate([c.values() for c in results_for_cls], list(results_for_cls[0].keys())), '\n')


    def at_k_metrics(self, k):
        assert k>0 and k<=self.n_cls, 'k should not exceed the number of categories or less than 1.'
        prec_k_list = []
        recall_k_list = []
        f1_k_list = []
        for i in range(self.n_sample):
            idx = np.argsort(self.pred_prob[i], axis=0)[-k:]
            top_k = np.zeros(self.n_cls)
            top_k[idx] = 1
            # Precision@k
            intersec = np.logical_and(self.ground_truth[i],top_k)
            prec_ki = sum(intersec)/k
            prec_k_list.append(prec_ki)

            # Recall@k
            len_gt = sum(self.ground_truth[i])
            if len_gt>k:
                recall_ki = sum(intersec)/k
            else:
                recall_ki = sum(intersec)/len_gt
            recall_k_list.append(recall_ki)

            # F1@k
            if prec_ki + recall_ki == 0:
                f1_ki = 0
            else:
                f1_ki = 2 * prec_ki * recall_ki / (prec_ki + recall_ki)
            f1_k_list.append(f1_ki)

        prec_k = round(sum(prec_k_list)/len(prec_k_list), self.dec_place)
        recall_k = round(sum(recall_k_list)/len(recall_k_list), self.dec_place)
        f1_k = round(sum(f1_k_list)/len(f1_k_list), self.dec_place)
        return prec_k, recall_k, f1_k
    
    def print_at_k_metrics(self):
        k_list = []
        k_range = range(1, self.at_k_value+1)
        for k in k_range:
            prec_k, recall_k, f1_k = self.at_k_metrics(k)
            k_list.append({
                'K':k,
                'Precision@k': prec_k,
                'Recall@k': recall_k,
                'F1@k': f1_k
            })
        print('@K Metrics:')
        print(tabulate([k.values() for k in k_list], list(k_list[0].keys())), '\n')

    def mismatch_id(self, i):
        class_name = self.target_names[i]
        incorrect = np.where(self.ground_truth[:,i]!=self.pred_lb[:,i])[0]
        incorrect = str(incorrect)
        return {
            'Class': class_name,
            'Index': incorrect
        }

    def print_per_class_exam(self):
        cls_incorrect = []
        for i in range(self.n_cls):
            cls_incorrect.append(self.mismatch_id(i))
        print('Incorrect predictions:')
        print(tabulate([c.values() for c in cls_incorrect], list(cls_incorrect[0].keys())), '\n')
        
        
    def print_posts(self, i):
        print(self.x_test[i]['id'])
        print(self.x_test[i]['title'])
        print(self.x_test[i]['description'])
        print(self.target_names)
        print('Ground Truth: ', self.ground_truth[i].astype(int))
        print('Predicted as: ', self.pred_lb[i].astype(int))
        print('Props:', (np.around(self.pred_prob[i].astype(float),3)))
        
        return
        
if __name__ == "__main__":
    if args.post:
        evaluator = Evaluation(args.k, args.r, args.pred_path, args.gt_path, args.post)
    else:
        evaluator = Evaluation(args.k, args.r, args.pred_path, args.gt_path)
    if args.i!=-1: evaluator.print_posts(args.i)
    if args.error: evaluator.print_per_class_exam()
    if args.overall: evaluator.print_overall_metrics()
    if args.perclass: evaluator.print_per_class_results()
    if args.atk: evaluator.print_at_k_metrics()
    