from pickle import FALSE
from re import escape
from numpy import random
from torch.utils.data import DataLoader, Dataset
import numpy as np
import multiprocessing as mp
from tqdm import trange, tqdm
import time

class MyData(Dataset):
    def __init__(self, reviewData, metaData, neg_sample_num, max_query_len, max_review_len, time_num, weights = True, 
                recent_item_len=5, recent_user_len=5):
        
        # userID and the one-hot id
        self.id2user = dict()
        self.user2id = dict()
        
        # productID and one-hot id
        self.id2product = dict()
        self.product2id = dict()
        
        # [asin, query]
        self.product2query = dict()
        
        # query
        self.word2id = dict()
        self.id2word = dict()
        
        self.userReviews = dict()
        self.userReviewsCount = dict()
        self.userReviewsCounter = dict()
        self.userReviewsTest = dict()
        
        self.max_review_len = max_review_len
        self.max_query_len = max_query_len
        self.neg_sample_num = neg_sample_num
        self.recent_item_len = recent_item_len
        self.recent_user_len = recent_user_len
        
        self.time_num = time_num
        self.time_data = []
        
        self.init_dict(reviewData, metaData)
        
        self.neg_item = [i for i in range(self.productNum)]
        self.neg_item_weight = [1e-2 for _ in range(self.productNum)]

        self.neg_user = [i for i in range(self.userNum)]
        self.neg_user_weight = [1e-2 for _ in range(self.userNum)]

        self.neg_word = [i for i in range(self.wordNum)]
        self.neg_word_weight = [1e-2 for _ in range(self.wordNum)]

        self.train_data = []
        self.test_data = []
        self.eval_data = []

        self.init_dataset(reviewData, weights)
        

    def init_dict(self, reviewData, metaData):
        for i in range(self.time_num):
            self.time_data.append([])
        
        uid = 0
        us = set(reviewData['reviewerID'])
        pr = set()
        words = set()
        group_data = reviewData.groupby('reviewerID', sort=False)
        
        for u in tqdm(us):
            # remove the user with 2 records at most
            user_data = group_data.get_group(u)
            asins = list(user_data['asin'])

            if(len(asins) <=2 ):
                continue

            self.id2user[uid] = u
            self.user2id[u] = uid
            
            # get the bought list of every user
            pr.update(asins)
            self.userReviews[uid] = asins

            # set the last one as test
            self.userReviewsTest[uid] = asins[-1]

            words.update(set(' '.join(list(user_data['reviewText'])).split()))
            uid += 1

        self.userNum = uid
        
        pid = 0
#         words = set()
        for p in tqdm(pr):
            try:
                '''
                判断这个product是否有query
                '''
                if (len(metaData.loc[p]['query']) > 0):
                    self.product2query[p] = metaData.loc[p]['query']
                    words.update(' '.join(metaData.loc[p]['query']).split(' '))
            except:
                pass
            self.id2product[pid] = p
            self.product2id[p] = pid
            pid += 1
            
        self.productNum = pid
        self.queryNum = len(self.product2query)
        
        wi = 0
        self.word2id['<pad>'] = wi
        self.id2word[wi] = '<pad>'
        wi += 1
        for w in tqdm(words):
            if(w==''):
                continue
            self.word2id[w] = wi
            self.id2word[wi] = w
            wi += 1
            
        self.wordNum = wi


    def init_dataset(self, reviewData,weights=True):
        self.data_X = []
        
        # save the dict of data to save time
        time_data = reviewData.groupby('timeBin')
        recent_item = dict()
        recent_user = dict()

        for index_, data in time_data:

            recent_item[index_] = data.groupby('reviewerID')
            recent_user[index_] = data.groupby('asin')

        for r in trange(len(reviewData)):
            rc = reviewData.iloc[r]

            try:
                uid = self.user2id[rc['reviewerID']]
                pid_pos = self.product2id[rc['asin']]
                time_bin_pos = int(rc['timeBin'])
            except:
                continue

            # get the query
            try:
                q_text_array_pos = self.product2query[rc['asin']]
                
            except:
                continue
            
            # exact the recent items and recent users
            recent_items = recent_item[time_bin_pos].get_group(rc['reviewerID'])
            recent_users = recent_user[time_bin_pos].get_group(rc['asin'])

            recent_bought_items = []
            recent_bought_users = []

            for i in range(len(recent_items),0,-1):
                try:
                    temp_pid = self.product2id[recent_items.iloc[i-1]['asin']]
                    recent_bought_items.append(temp_pid)
                except:
                    continue
                if len(recent_bought_items) >= self.recent_item_len:
                    break

            recent_bought_items = list(reversed(recent_bought_items))
            recent_items_length = len(recent_bought_items)
            for i in range(self.recent_item_len-len(recent_bought_items)):
                recent_bought_items.append(0)
            


            for i in range(len(recent_users),0,-1):
                try:
                    temp_uid = self.user2id[recent_users.iloc[i-1]['reviewerID']]
                    recent_bought_users.append(temp_uid)
                except:
                    continue
                if len(recent_bought_users) >= self.recent_user_len:
                    break
            
            recent_bought_users = list(reversed(recent_bought_users))
            recent_users_length = len(recent_bought_users)
            for i in range(self.recent_user_len-len(recent_bought_users)):
                recent_bought_users.append(0)


            # exact the query and append into the training data
            num = 0
            for qi in q_text_array_pos:
                qids_pos, len_pos = self.trans_to_ids(qi, self.max_query_len)
                # add the training data
                self.data_X.append((uid, pid_pos, qids_pos, len_pos, time_bin_pos, recent_bought_items, recent_items_length, recent_bought_users, recent_users_length))
                self.neg_item_weight[pid_pos] += 1
                self.neg_user_weight[uid] += 1
                num += 1
            
            if uid not in self.userReviewsCount.keys():
                self.userReviewsCount[uid] = num
                self.userReviewsCounter[uid] = num
            else:
                self.userReviewsCount[uid] += num
                self.userReviewsCounter[uid] += num

        for r in self.data_X:

            if self.userReviewsCount[r[0]] > 2:
                t = self.userReviewsCounter[r[0]]
                if (t == 0):
                    continue
                elif (t == 2):  # the second last one 
                    self.eval_data.append(r)
                elif (t == 1):  # the last one
                    self.test_data.append(r)
                else:
                    self.train_data.append(r)
                    self.time_data[r[4]].append(r)
                self.userReviewsCounter[r[0]] -= 1


        # get the negative sampling probability
        user_sample_pro = np.array(self.neg_user_weight)
        self.user_sample_pro = user_sample_pro/user_sample_pro.sum()
        item_sample_pro = np.array(self.neg_item_weight)
        self.item_sample_pro = item_sample_pro/item_sample_pro.sum()


    def trans_to_ids(self, query, max_len):
        query = query.split(' ')
        qids = []
        for w in query:
            if w == '':
                continue
            try:
                qids.append(self.word2id[w])
            except:
                continue
            # 需要统计词频
            self.neg_word_weight[self.word2id[w]] += 1
        for _ in range(len(qids), max_len):
            qids.append(self.word2id['<pad>'])
        return qids, len(query)
    
    
    def __getitem__(self, i):
        
        pos = self.train_data[i]
        neg_users = self.neg_user_sampling(pos[0])
        neg_items = self.neg_item_sampling(pos[1])
        neg_words = self.neg_word_sampling(pos[2], pos[3])
        return pos, neg_users, neg_items, neg_words

    def get_time_data(self, time_bin, i):
        pos = self.time_data[time_bin][i]
        neg_users = self.neg_user_sampling(pos[0])
        neg_items = self.neg_item_sampling(pos[1])
        return pos, neg_users, neg_items


    def getTestItem(self, i):
        return self.test_data[i]

        
    def __len__(self):
        return len(self.train_data)

    def neg_item_sampling(self, pos_pid_):
        '''
        sampling the negative sample for the positive sample
        '''
        neg_samples = []
        for _ in range(self.neg_sample_num):
            sample_one = np.random.choice(self.neg_item, p=self.item_sample_pro)
            while(sample_one == pos_pid_):
                sample_one = np.random.choice(self.neg_item, p=self.item_sample_pro)
            neg_samples.append(sample_one)

        return neg_samples
    
    def neg_user_sampling(self, pos_uid_):
        '''
        sampling the negative sample for the positive sample
        '''
        neg_samples = []
        for _ in range(self.neg_sample_num):
            sample_one = np.random.choice(self.neg_user, p=self.user_sample_pro)
            while(sample_one == pos_uid_):
                sample_one = np.random.choice(self.neg_user, p=self.user_sample_pro)
            neg_samples.append(sample_one)

        return neg_samples

    def neg_word_sampling(self, pos_query, pos_len):
        '''
        sampling the negative sample for the words in the query
        '''

        neg_samples = []
        for _ in range(self.neg_sample_num):
            sample_one = np.random.choice(self.neg_user, p=self.user_sample_pro)
            while(sample_one in pos_query):
                sample_one = np.random.choice(self.neg_user, p=self.user_sample_pro)
            neg_samples.append(sample_one)
        
        return neg_samples