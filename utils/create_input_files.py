import numpy as np
import pandas as pd
import sys, getopt
import pickle
import toml
import os
from utils.bm25Vectorizer import bm25
from utils.utils import get_errorper
from sklearn.preprocessing import LabelEncoder

########################################################################################

def load_config(config_file):
    config = toml.load(config_file)
    print(config)
    return config

########################################################################################

def read_data(config):
    with open(config['data_file'], 'rb') as handle:
        data = pickle.load(handle)
    yplus = data[[config['label_col'], config['bg_col']]]
    vocab_file = config.get('vocab', None)

    if vocab_file != None:
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = None
    
    #if not os.path.exists('data/features.pickle'):
    if not os.path.exists('data/features2.pickle'):
        sparse_bm25, _, _, _ = bm25(data['text'], min_df=10, vocab=vocab)
    else:
        sparse_bm25 = None
    return sparse_bm25, yplus

########################################################################################

def add_selection_columns(yplus, config, seed):
    label = config['label_col']

    ############################################################
    # add columns indicating records that belong to gold sets
    ############################################################
    
    sizegold = config['sizegold']
    sizegold.sort(reverse=True)
    gold = yplus.copy()

    for size in sizegold:
        gold = gold.groupby(label, group_keys=False).apply(lambda group_df: group_df.sample(size, random_state=seed))
        yplus[f'gold{size}'] = 0
        yplus.loc[gold.index, f'gold{size}'] = 1

    ############################################################
    # add columns containing new (manipulated) labels
    # records are chosen from the records not in the largest gold set
    ############################################################

    index_noisy = yplus[yplus[f'gold{sizegold[0]}']==0].index
    yplus = create_sample(yplus, index_noisy, config)
    
    errortypes = config['errortypes']
    errorper = config['errorper']
    bg_col = config['bg_col']
    PrCl = [f'PC{c}' for c in yplus[bg_col].unique()]
    # PrCl = config['PrCl']
    matrix = {}
    K = yplus[label].nunique()

    for err in errortypes:
        errorper = get_errorper(config, err)
        for per in errorper:
            label_new = f'label_{per}{err}'
            per_str = '' if per == '' else str(100 + per)[1:]
            yplus[label_new] = yplus[label] # new sbi == actual sbi.

            for C in PrCl:
                # LSZU: pad veranderen?
                file_name = f'V3/dry_beans/Matrices_PC/{"" if per=="" else per_str+"perc_"}{C}.csv'
                labels = list(pd.read_csv(file_name, sep=';').columns[1:])
                matrix[int(C[-1])] = np.asarray(pd.read_csv(file_name, index_col=[0], sep=';'))

            label_to_index = {label: idx for idx, label in enumerate(labels)}
            
            for i in index_noisy:
                C = yplus.loc[i, bg_col]
                k = yplus.loc[i, label]
                k_index = label_to_index[k]
                yplus.loc[i, label_new] = np.random.choice(K, p=matrix[C][k_index,:]) 
                # Voor droge bonen: k is nu de naam van de seed... we hebben een labelencoder nodig
    return yplus

########################################################################################

def create_sample(yplus, index_noisy, config):
    mapping_file = config.get('mapping_file', None)

    if mapping_file != None:
        mapping = pd.read_csv(mapping_file, sep=';')
        bg_cols = mapping.columns
        yplus = yplus.merge(mapping, on=bg_cols[0], how='left')
    else:
        bg_cols = [config['bg_col']]

    sample = yplus.loc[index_noisy]
    for bg_col in bg_cols:
        yplus[f'sample_{bg_col}'] = 0
        sample = sample.groupby(bg_col, group_keys=False).apply(lambda group_df: group_df.sample(config['n_sample']))
        yplus.loc[sample.index, f'sample_{bg_col}'] = 1
    
    return yplus

########################################################################################

def start(config):
    seeds = config['seeds']
    features, y = read_data(config)

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        yplus = add_selection_columns(y, config, seed)
        yplus.to_csv(f'data/labels_Seed{i+1}.csv', sep=';', index=False)
    
    if features != None:
        #with open('data/features.pickle', 'wb') as f:
        with open('data/features2.pickle', 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

########################################################################################

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:", ["config="])
    except:
        print("usage: python create_input_files.py -c <filename>")
        sys.exit(2)

    if len(opts) > 1:
        print("usage: python create_input_files.py -c <filename>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ["-c", "--config"]:
            try:
                config = load_config(arg)
                print(config)
            except ValueError as e:
                print(e)
                print(f'file {arg} not found')
                sys.exit(2)
        else:
            print("usage: create_input_files.py -c <filename>")
            sys.exit(2)
        
    start(config)
