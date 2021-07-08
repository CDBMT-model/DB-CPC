import gzip
import json
from dateutil.parser import parse as pt
import pandas as pd
import numpy as np
import random
from tqdm import tqdm, trange
import os, sys
import nltk
import re
import pickle
import multiprocessing as mp
import argparse

sys.path.append(os.getcwd())
import config
from scrips.mydata import MyData

import warnings
warnings.filterwarnings("ignore")


def get_meta_data(file_name):
    path = os.path.join(config.main_meta_path, 'meta_'+file_name+'.json.gz')
    meda_g = gzip.open(path)
    
    rr = []
    for l in meda_g:
        l = l.decode()[:-1]
        try:
            rr.append(eval(l))
        except:
            print(l.decode()[:-1].replace('\'', '\"'))
    values = dict()
    ks = ['asin', 'categories']
    for k in ks:
        values[k] = []
    for i in range(len(rr)):
        try:
            for k in ks:
                values[k].append(rr[i][k])
        except:
            print(rr[i])
            
    meta_datas = pd.DataFrame(values)
    meta_datas.set_index('asin', inplace=True)
    
    return meta_datas



def get_review_data(file_name):
    '''
    get the review data
    input: file_name(str)
    output: data(DataFrame)
    '''
    path = os.path.join(config.main_review_path, 'reviews_'+file_name+'_5.json.gz')
    g = gzip.open(path)
    
    reviews = []
    for l in g:
        reviews.append(json.loads(l.decode()[:-1]))
        
    values = dict()
    ks = ['reviewerID', 'asin', 'reviewText', 'reviewTime', 'unixReviewTime']
    for k in ks:
        values[k] = []
    for i in range(len(reviews)):
        try:
            for k in ks:
                values[k].append(reviews[i][k])
        except:
            print(reviews[i])
    review_datas = pd.DataFrame(values) # get the dataframe of dataset
    return review_datas


def process_review_data(data):
    
    stop_path = config.stop_file
    stop_df = pd.read_csv(stop_path, header=None, names=['stopword'])
    stop_words = set(stop_df['stopword'].unique())

    def help_f_cut_stop_word(x):
        '''
        remove the special symbol
        '''
        x = x.lower()
        x = re.sub(r'([;\.~\!@\#\$\%\^\&\*\(\(\)_\+\=\-\[\]\)/\|\'\"\?<>,`\\])','',x) #将特殊的符号去掉。sub的作用是替换符合的字符
        ss = ""
        words = x.split(' ')
        
        #stopwords = nltk.corpus.stopwords.words('english') + list(';.~!@#$:%^&*(()_+=-[])/|\'\"?<>,`\\1234567890')
        
        for w in words:
            if (w in stop_words):
                pass
            else:
                ss += ' ' + w

        return ss.lower().strip()
    
    def cutWord(x):
        '''
        cut the too long words
        '''
        if (len(x.split(' ')) > mean_review_len):
            x = x[:mean_review_len]
        return x
    
    data['reviewText'] = data['reviewText'].map(help_f_cut_stop_word)
    print('remove the special symbol')

    r_lens = []
    for i in data['reviewText']:
        r_lens.append(len(i.split(' ')))
    mean_review_len = int(sum(r_lens)/len(r_lens))
    print('get the mean len')

    data['reviewText'] = data['reviewText'].map(cutWord)
    data = data.sort_values(by='unixReviewTime', ascending=True)
    
    # add the timestap
    data['timeBin'] = None
    time_scope = [j for j in range(1997,2016)]
    dataBins = [pt(str(i)+"-1-1").timestamp() for i in time_scope]
    temporal_datas = [pd.DataFrame(columns=('reviewerID', 'asin', 'reviewText', 'reviewTime', 'unixReviewTime')) 
                      for _ in range(len(time_scope)-1)]
    for t in range(0,len(dataBins)-1):
        temporal_datas[t] = data[(data['unixReviewTime'] >= dataBins[t]) & (data['unixReviewTime'] < dataBins[t+1])]
        temporal_datas[t].loc[:, 'timeBin'] = t
        data.loc[(data['unixReviewTime'] >= dataBins[t]) & (data['unixReviewTime'] < dataBins[t+1]), 'timeBin'] = t
        
    return data, temporal_datas, mean_review_len



def get_query(x):
    qs = list()
    for sub_cat_list in x:
        if (len(sub_cat_list) <= 1):
            continue
        qs.append(sub_cat_list)
    '''
    remove duplicate
    '''
    finalQs = []
    
    for q in qs:
        
        Q_words = ' '.join(q).lower().replace(' & ', ' ').replace(',', '').strip().split(' ')
        finalQ = ''
        words = set()
        for i in range(len(Q_words)-1, -1, -1):
            if (Q_words[i] not in words):
                finalQ = Q_words[i] + ' ' + finalQ
                words.add(Q_words[i])
        finalQs.append( finalQ.strip())
    return finalQs



def process_meta_data(meta_datas):
    '''
    process the meta data
    '''
    meta_datas['query'] = meta_datas['categories'].map(get_query)
    return meta_datas

def get_max_query_len(meta_datas):
    q_lens = []
    for i in meta_datas['query']:
        for q in i:
            q_lens.append(len(q.split(' ')))
    max_query_len = max(q_lens)

    return max_query_len

def get_review_max(review_datas):
    re_lens = []

    for i in trange(len(review_datas)):
        re_lens.append(len(review_datas.iloc[i]['reviewText'].split(' ')))
    
    return max(re_lens)

def get_query_max(meta_datas):
    q_lens = []

    for i in trange(len(meta_datas)):
        for one_query in meta_datas.iloc[i]['query']:
            q_lens.append(len(one_query.split(' ')))
            #print(q_lens[-1])
    
    return max(q_lens)

def main():

    #data_name = 'Toys_and_Games'
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
        type=int,
        default=0,
        help="choose the dataset")
    FLAGS = parser.parse_args()    
    dataset_type = FLAGS.dataset
    if dataset_type == 0:
        data_name = "Toys_and_Games"
    elif dataset_type == 1:
        data_name = "Clothing_Shoes_and_Jewelry"
    elif dataset_type == 2:
        data_name = "Cell_Phones_and_Accessories"
    elif dataset_type == 3:
        data_name = "Electronics"
    elif dataset_type == 4:
        data_name = "Beauty"
    else:
        data_name = "Toys_and_Games"

    if not os.path.exists(os.path.join(config.processed_path, 'review_datas_time_bin_word'+data_name+'.bin')):
        review_datas = get_review_data(data_name)
        print("get the review data")
        review_datas, temporal_datas, max_review_len = process_review_data(review_datas)
        print("process the review data successfully")
        print(max_review_len)
        meta_datas = get_meta_data(data_name)
        print("get the meta data")
        meta_datas = process_meta_data(meta_datas)
        print("process the meta data successfully")
        max_query_len = get_max_query_len(meta_datas)
        print(max_query_len)

        if not os.path.exists(config.processed_path):
            os.mkdir(config.processed_path)
        with open(os.path.join(config.processed_path, 'review_datas_time_bin_word'+data_name+'.bin'), 'wb+') as f:
            pickle.dump(review_datas,f)
        with open(os.path.join(config.processed_path, 'temporal_datas_time_bin_word'+data_name+'.bin'), 'wb+') as f:
            pickle.dump(temporal_datas,f)
        with open(os.path.join(config.processed_path, 'meta_datas_time_bin_word'+data_name+'.bin'), 'wb+') as f:
            pickle.dump(meta_datas,f)

        print("save all data")
    else:
        with open(os.path.join(config.processed_path, 'review_datas_time_bin_word'+data_name+'.bin'), 'rb') as f:
            review_datas = pickle.load(f)
        print('get the review data')
        max_review_len = get_review_max(review_datas)
        
        with open(os.path.join(config.processed_path, 'temporal_datas_time_bin_word'+data_name+'.bin'), 'rb') as f:
            temporal_datas = pickle.load(f)
        print('get the temporal data')

        with open(os.path.join(config.processed_path, 'meta_datas_time_bin_word'+data_name+'.bin'), 'rb') as f:
            meta_datas = pickle.load(f)
        print('get the meta data')
        max_query_len = get_query_max(meta_datas)

    # ----------- save the dataset -----------------
    print('begin to process the dataset')
    # print(max_review_len)
    # print(max_query_len)
    data_loader = MyData(review_datas, meta_datas, 5, max_query_len, max_review_len, len(temporal_datas), weights = True)
    # data_loader = MyData(review_datas, meta_datas, 5, 12, 54, len(temporal_datas), weights = True)
    with open(os.path.join(config.processed_path, 'dataset_time_'+data_name+'.bin'), 'wb+') as f:
        pickle.dump(data_loader,f)

if __name__ == "__main__":
    main()
