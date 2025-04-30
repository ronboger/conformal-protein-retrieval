import csv
from CLEAN.utils import *
from CLEAN.distance_map import *
from CLEAN.evaluate import *
from CLEAN.model import LayerNormNet
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, average_precision_score
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import pickle

def get_true_labels_test(file_name, test_idx: None):
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter='\t')
    all_label = set()
    true_label_dict = {}
    header = True
    count = 0
    for row in csvreader:
        # don't read the header
        if header is False:
            count += 1
            true_ec_lst = row[1].split(';')
            true_label_dict[row[0]] = true_ec_lst
            for ec in true_ec_lst:
                if test_idx is not None and count - 1 in test_idx:
                    all_label.add(ec)
                elif test_idx is None: ## add all EC labels
                    all_label.add(ec)
                else:
                    continue
        if header:
            header = False
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    if test_idx is not None:
        true_label = [true_label[i] for i in test_idx]
    return true_label, all_label


def infer_conformal(train_data, test_data, thresh, report_metrics=False, 
                 pretrained=True, model_name=None, test_idx=None, name_id="1"):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data + name_id
    if test_idx is None:
        idx = [i for i in range(len(id_ec_test))]
        write_conformal_choices(eval_df, out_filename, threshold=thresh, test_idx=idx)
    else:
        write_conformal_choices(eval_df, out_filename, threshold=thresh, test_idx=test_idx)
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_conformal')
        pred_probs = get_pred_probs(out_filename, pred_type='_conformal')
        true_label, all_label = get_true_labels_test('./data/' + test_data, test_idx=test_idx if test_idx is not None else None)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print("############ EC calling results using conformal calibration on randomly shuffled test set ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)
    return pre, rec, f1, roc


## In theory, we should be able to use the lambda we find on the raw eval distance map,
## slice the test set out of it, and pass it into a small method, infer_confromal
## that will take in the eval_df and the lambda, and write the choices using this method,
## then report all the metrics we want.
def write_conformal_choices(df, csv_name, threshold, test_idx: list):
    """
    df: dataframe containing the distances between the test set and the EC centroids
    csv_name: name of the csv file to write the choices to
    threshold: threshold to use for the choices (euclidean distance by default, so <=)
    test_idx: list of indices of the test set within the dataframe. this is how we splice the columns
            to get the ones we want to test on, not the ones calibrated on.
    """
    out_file = open(csv_name + '_conformal.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    dists = []
    for col in df.iloc[:, test_idx].columns:
        ec = []
        dist_lst = []
        ## grsb EC numbers bounded by threshold
        smallest_dists_thresh = df[col][df[col] <= threshold]
        for i in range(len(smallest_dists_thresh)):
            EC_i = smallest_dists_thresh.index[i]
            dist_i = smallest_dists_thresh[i]
            dist_str = "{:.4f}".format(dist_i)
            dist_lst.append(dist_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        dists.append(dist_lst)
        csvwriter.writerow(ec)
    return dists


## Below code is taken from CLEAN/evaluate.py, but modified to take in the test_idx and only eval on that

def infer_maxsep(train_data, test_data, report_metrics = False, 
                 pretrained=True, model_name=None, gmm = None, test_idx=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data + "test_idx"
    if test_idx is None:
        idx = [i for i in range(len(id_ec_test))]
        write_max_sep_choices_test(eval_df, out_filename, gmm=gmm, test_idx=idx)
    else:
        write_max_sep_choices_test(eval_df, out_filename, gmm=gmm, test_idx=test_idx)
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels_test('./data/' + test_data, test_idx=test_idx if test_idx is not None else None)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print("############ EC calling results using maximum separation on randomly shuffled test set ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)
    return pre, rec, f1, roc


def write_max_sep_choices_test(df, csv_name, test_idx, first_grad=True, use_max_grad=False, gmm = None):
    out_file = open(csv_name + '_maxsep.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    for col in df.iloc[:, test_idx].columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            if gmm != None:
                gmm_lst = pickle.load(open(gmm, 'rb'))
                dist_i = infer_confidence_gmm(dist_i, gmm_lst)
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return

def infer_pvalue(train_data, test_data, p_value = 1e-5, nk_random = 20, 
                 report_metrics = False, pretrained=True, model_name=None, test_idx=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
        
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    rand_nk_ids, rand_nk_emb_train = random_nk_model(
        id_ec_train, ec_id_dict_train, emb_train, n=nk_random, weighted=True)
    random_nk_dist_map = get_random_nk_dist_map(
        emb_train, rand_nk_emb_train, ec_id_dict_train, rand_nk_ids, device, dtype)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data
    #write_pvalue_choices( eval_df, out_filename, random_nk_dist_map, p_value=p_value)

    if test_idx is None:
        idx = [i for i in range(len(id_ec_test))]
        write_pvalue_choice_test(eval_df, out_filename, random_nk_dist_map, p_value = p_value, test_idx=idx)
    else:
        write_pvalue_choice_test(eval_df, out_filename, random_nk_dist_map, p_value = p_value, test_idx=test_idx)

    # optionally report prediction precision/recall/...
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_pvalue')
        pred_probs = get_pred_probs(out_filename, pred_type='_pvalue')
        true_label, all_label = get_true_labels_test('./data/' + test_data, test_idx=test_idx if test_idx is not None else None)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print(f'############ EC calling results using random '
        f'chosen {nk_random}k samples ############')
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)  
    return pre, rec, f1, roc
    

def write_pvalue_choice_test(df, csv_name, random_nk_dist_map, p_value=1e-5, test_idx = None):
    out_file = open(csv_name + '_pvalue.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    nk = len(random_nk_dist_map.keys())
    threshold = p_value*nk
    for col in df.iloc[:, test_idx].columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            # find all the distances in the random nk w.r.t. EC_i
            # then sorted the nk distances
            rand_nk_dists = [random_nk_dist_map[rand_nk_id][EC_i]
                             for rand_nk_id in random_nk_dist_map.keys()]
            rand_nk_dists = np.sort(rand_nk_dists)
            # rank dist_i among rand_nk_dists
            dist_i = smallest_10_dist_df[i]
            rank = np.searchsorted(rand_nk_dists, dist_i)
            if (rank <= threshold) or (i == 0):
                dist_str = "{:.4f}".format(dist_i)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
            else:
                break
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


