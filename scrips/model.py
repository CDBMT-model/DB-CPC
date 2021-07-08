from numpy.lib.twodim_base import mask_indices
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
product search model
'''
'''
product search model
'''
class DBCPC(nn.Module):
    def __init__(self, user_size, item_size, word_size, embedding_dim,\
                 max_query_len, max_review_len, batch_size, time_num,\
                 neg_num=5,sample_num=1,transfer_hidden_dim=100,\
                 max_private_len = 3, max_public_len = 10,\
                 item_log_len = 3, user_log_len = 3, \
                 recent_item_max = 5, recent_user_max = 5, \
                 sigma_parameter=1e0, kl_parameter=1e0,\
                 user_cpc_parameter=1e-2, item_cpc_parameter=1e-2, word_cpc_parameter=1e-2,\
                 recent_item_parameter=1e-2, recent_user_parameter=1e-2,\
                 device=torch.device('cpu')):
        super(DBCPC, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.word_size = word_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_query_len = max_query_len
        self.max_review_len = max_review_len
        self.item_log_len = item_log_len
        self.user_log_len = user_log_len
        self.recent_item_max = recent_item_max
        self.recent_user_max = recent_user_max
        self.sample_num = sample_num
        self.transfer_hidden_dim = transfer_hidden_dim
        self.max_private_len = max_private_len
        self.max_public_len = max_public_len
        self.kl_parameter = kl_parameter
        self.sigma_parameter = sigma_parameter
        self.user_cpc_parameter = user_cpc_parameter
        self.item_cpc_parameter = item_cpc_parameter
        self.word_cpc_parameter = word_cpc_parameter
        self.recent_item_parameter = recent_item_parameter
        self.recent_user_parameter = recent_user_parameter
        self.device = device
        self.neg_num = neg_num
        self.time_num = time_num
        
        self.esp = 1e-10
        
        
        
        self.time_embdding = nn.Embedding(self.time_num, self.embedding_dim)
        self.time2mean_u = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2mean_i = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2mean_w = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2std_i = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2std_u = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2std_w = nn.Linear(self.embedding_dim*2, self.embedding_dim)

        # CPC
        self.regre_u_mean = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.regre_u_std = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.predict_u_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.predict_u_std = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.regre_i_mean = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.regre_i_std = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.predict_i_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.predict_i_std = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.regre_w_mean = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.regre_w_std = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.predict_w_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.predict_w_std = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Contrastive learning for recent behaviour
        self.regre_recent_item_mean = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.regre_recent_item_std = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.predict_recent_item_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.predict_recent_item_std = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.regre_recent_user_mean = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.regre_recent_user_std = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.predict_recent_user_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.predict_recent_user_std = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.user_mean = nn.Embedding(self.user_size, self.embedding_dim, _weight=torch.ones(self.user_size, self.embedding_dim))
        self.user_std = nn.Embedding(self.user_size, self.embedding_dim, _weight=torch.zeros(self.user_size, self.embedding_dim))
        
        self.item_mean = nn.Embedding(self.item_size, self.embedding_dim, _weight=torch.ones(self.item_size, self.embedding_dim))
        self.item_std = nn.Embedding(self.item_size, self.embedding_dim, _weight=torch.zeros(self.item_size, self.embedding_dim))
        
        
        self.wordEmbedding_mean = nn.Embedding(self.word_size, self.embedding_dim, padding_idx=0, _weight=torch.ones(self.word_size, self.embedding_dim))
        self.wordEmbedding_std = nn.Embedding(self.word_size, self.embedding_dim, padding_idx=0, _weight=torch.zeros(self.word_size, self.embedding_dim))
        self.queryLinear = nn.Linear(self.embedding_dim, self.embedding_dim)

        # prior
        self.transfer_linear_u = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_i = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_ni = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)

        self.transfer_mean_u = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_i = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_ni = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)

        self.transfer_std_u = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_i = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_ni = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)

        # contrastive module
        self.private_mean_transfer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.private_std_transfer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.public_mean_transfer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.public_std_transfer = nn.Linear(self.embedding_dim, self.embedding_dim)

        # word contrastive module
        self.review_user_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.review_user_std = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.review_item_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.review_item_std = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    '''
    (uid, pid_pos, qids_pos, len_pos, time_bin_pos)
    [( uid, pid, qids_neg, len_neg, time_bin_pos),..,( uid, pid, qids_neg, len_neg, time_bin_pos)]*neg_sample_num
    '''
    
    def forward(self, user, item_pos, query, query_len, times, recent_items, recent_items_len, recent_users, recent_users_len, user_neg, items_neg, word_neg):
        self.batch_size = user.shape[0]
        '''
        time embedding
        '''
        time_laten = self.time_embdding(times+torch.tensor(1).to(self.device)).squeeze(1)
        pri_time_laten =self.time_embdding(times)
        
        '''
        user
        '''
        user_mean_emb = self.user_mean(user).squeeze(1) # (batch, out_size)
        user_mean_pri = self.time2mean_u(torch.cat([user_mean_emb, pri_time_laten], 1))
        user_mean = self.time2mean_u(torch.cat([user_mean_emb, time_laten], 1))
        
        user_std_emb = self.user_std(user).squeeze(1) # (batch, out_size)
        user_std_pri = self.time2std_u(torch.cat([user_std_emb, pri_time_laten], 1)).mul(0.5).exp()
        user_std = self.time2std_u(torch.cat([user_std_emb, time_laten], 1)).mul(0.5).exp()
        
        '''
        query
        '''
        word_mean_emb = self.wordEmbedding_mean(query)
        word_std_emb = self.wordEmbedding_std(query)
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_query_len-int(i.item())) for i in query_len]).unsqueeze(2).to(self.device)
        
        query = word_mean_emb.mul(len_mask)
        query = query.sum(dim=1).div(query_len.unsqueeze(1).float())
        query = self.queryLinear(query).tanh()
       
        
        '''
        pos product
        '''
        item_mean_emb = self.item_mean(item_pos).squeeze(1) # (batch, out_size)
        item_mean_pos_pri = self.time2mean_i(torch.cat([item_mean_emb, pri_time_laten], 1))
        item_mean_pos = self.time2mean_i(torch.cat([item_mean_emb, time_laten], 1))
        
        item_std_emb = self.item_std(item_pos).squeeze(1) # (batch, out_size)
        item_std_pos_pri = self.time2std_i(torch.cat([item_std_emb, pri_time_laten], 1)).mul(0.5).exp()
        item_std_pos = self.time2std_i(torch.cat([item_std_emb, time_laten], 1)).mul(0.5).exp()

        
        '''
        neg product
        '''
        items_mean_neg = self.item_mean(items_neg)# (batch, neg_sample_num, out_size)
        items_mean_neg_pri = self.time2mean_i(torch.cat([items_mean_neg, pri_time_laten.unsqueeze(1).expand_as(items_mean_neg)], 2))
        items_mean_neg = self.time2mean_i(torch.cat([items_mean_neg, time_laten.unsqueeze(1).expand_as(items_mean_neg)], 2))
        
        items_std_neg = self.item_std(items_neg)# (batch, neg_sample_num, out_size)
        items_std_neg_pri = self.time2std_i(torch.cat([items_std_neg, pri_time_laten.unsqueeze(1).expand_as(items_std_neg)], 2)).mul(0.5).exp()
        items_std_neg = self.time2std_i(torch.cat([items_std_neg, time_laten.unsqueeze(1).expand_as(items_std_neg)], 2)).mul(0.5).exp()

        '''
        contrastive learning for user
        '''
        time_emb = torch.stack([self.time_embdding(times - torch.tensor(i).to(self.device)) for i in range(times.min(), 0, -1)])
        all_user_mean = torch.stack([self.time2mean_u(torch.cat([user_mean_emb, time_emb[i]], 1)) for i in range(times.min())])
        user_context_mean = self.regre_u_mean(all_user_mean)[1]
        user_context_mean = self.predict_u_mean(user_context_mean).squeeze(0)

        all_user_std = torch.stack([self.time2mean_u(torch.cat([user_std_emb, time_emb[i]], 1)).mul(0.5).exp() for i in range(times.min())])
        user_context_std = self.regre_u_std(all_user_std)[1]
        user_context_std = self.predict_u_std(user_context_std).squeeze(0)

        user_mean_emb_neg = self.user_mean(user_neg)
        user_mean_neg = self.time2mean_u(torch.cat([user_mean_emb_neg, time_laten.unsqueeze(1).expand_as(user_mean_emb_neg)], 2))

        user_std_emb_neg = self.user_std(user_neg)
        user_std_neg = self.time2std_u(torch.cat([user_std_emb_neg, time_laten.unsqueeze(1).expand_as(user_std_emb_neg)], 2)).mul(0.5).exp()

        '''
        contrastive learning for item
        '''
        all_item_mean = torch.stack([self.time2mean_i(torch.cat([item_mean_emb, time_emb[i]], 1)) for i in range(times.min())])
        item_context_mean = self.regre_i_mean(all_item_mean)[1]
        item_context_mean = self.predict_i_mean(item_context_mean).squeeze(0)
        all_item_std = torch.stack([self.time2mean_i(torch.cat([item_std_emb, time_emb[i]], 1)).mul(0.5).exp() for i in range(times.min())])
        item_context_std = self.regre_i_std(all_item_std)[1]
        item_context_std = self.predict_i_std(item_context_std).squeeze(0)
        
        '''
        contrastive learning for word
        '''
        all_word_mean = torch.stack([self.time2mean_w(torch.cat([word_mean_emb, time_emb[i].unsqueeze(1).expand_as(word_mean_emb)], 2)) for i in range(times.min())])
        all_word_mean = all_word_mean.view(all_word_mean.size()[0] ,-1, all_word_mean.size()[-1])
        word_context_mean = self.regre_w_mean(all_word_mean)[1].squeeze(0)
        word_context_mean = word_context_mean.view(-1, self.max_query_len, self.embedding_dim)
        word_context_mean = self.predict_w_mean(word_context_mean)
        
        all_word_std = torch.stack([self.time2std_w(torch.cat([word_std_emb, time_emb[i].unsqueeze(1).expand_as(word_std_emb)], 2)).mul(0.5).exp() for i in range(times.min())])
        all_word_std = all_word_std.view(all_word_std.size()[0] ,-1, all_word_std.size()[-1])
        word_context_std = self.regre_w_std(all_word_std)[1].squeeze(0)
        word_context_std = word_context_std.view(-1, self.max_query_len, self.embedding_dim)
        word_context_std = self.predict_w_std(word_context_std)

        '''
        contrastive learning for recent items of users
        '''
        recent_items_len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.recent_item_max-int(i.item())) for i in recent_items_len]).unsqueeze(2).to(self.device)

        recent_items_mean_emb = self.item_mean(recent_items)
        recent_items_mean = self.time2mean_i(torch.cat([recent_items_mean_emb, time_laten.unsqueeze(1).expand_as(recent_items_mean_emb)], 2))
        recent_items_context_mean = self.regre_recent_item_mean(recent_items_mean.transpose(0,1))[0].transpose(0,1)
        recent_items_context_mean = recent_items_context_mean.mul(recent_items_len_mask).sum(dim=1).squeeze(1)
        recent_items_context_mean = self.predict_recent_item_mean(recent_items_context_mean)

        recent_items_std_emb = self.item_std(recent_items)
        recent_items_std = self.time2std_i(torch.cat([recent_items_std_emb, time_laten.unsqueeze(1).expand_as(recent_items_std_emb)], 2)).mul(0.5).exp()
        recent_items_context_std = self.regre_recent_item_std(recent_items_std.transpose(0,1))[0].transpose(0,1)
        recent_items_context_std = recent_items_context_std.mul(recent_items_len_mask).sum(dim=1).squeeze(1)
        recent_items_context_std = self.predict_recent_item_std(recent_items_context_std)

        '''
        contrastive learning for recent users of items
        '''
        recent_users_len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.recent_user_max-int(i.item())) for i in recent_users_len]).unsqueeze(2).to(self.device)

        recent_users_mean_emb = self.user_mean(recent_users)
        recent_users_mean = self.time2mean_u(torch.cat([recent_users_mean_emb, time_laten.unsqueeze(1).expand_as(recent_users_mean_emb)], 2))
        recent_users_context_mean = self.regre_recent_user_mean(recent_users_mean.transpose(0,1))[0].transpose(0,1)
        recent_users_context_mean = recent_users_context_mean.mul(recent_users_len_mask).sum(dim=1).squeeze(1)
        recent_users_context_mean = self.predict_recent_user_mean(recent_users_context_mean)

        recent_users_std_emb = self.user_std(recent_users)
        recent_users_std = self.time2std_u(torch.cat([recent_users_std_emb, time_laten.unsqueeze(1).expand_as(recent_users_std_emb)], 2)).mul(0.5).exp()
        recent_users_context_std = self.regre_recent_user_std(recent_users_std.transpose(0,1))[0].transpose(0,1)
        recent_users_context_std = recent_users_context_std.mul(recent_users_len_mask).sum(dim=1).squeeze(1)
        recent_users_context_std = self.predict_recent_user_std(recent_users_context_std)

        # positive words
        pos_word_mean_emb = self.time2mean_w(torch.cat([word_mean_emb, time_laten.unsqueeze(1).expand_as(word_mean_emb)], 2))
        pos_word_std_emb = self.time2std_w(torch.cat([word_std_emb, time_laten.unsqueeze(1).expand_as(word_std_emb)], 2)).mul(0.5).exp()
        

        # negative words
        word_mean_neg = self.wordEmbedding_mean(word_neg)
        neg_word_mean_emb = self.time2mean_w(torch.cat([word_mean_neg, time_laten.unsqueeze(1).expand_as(word_mean_neg)], 2))

        word_std_neg = self.wordEmbedding_std(word_neg)
        neg_word_std_emb = self.time2std_w(torch.cat([word_std_neg, time_laten.unsqueeze(1).expand_as(word_std_neg)], 2)).mul(0.5).exp()

        '''
        用户和product word的隐变量采样
        '''
        user_sample = self.reparameter(user_mean, user_std)
        product_sample = self.reparameter(item_mean_pos, item_std_pos)
        product_sample_neg = self.reparameter(items_mean_neg, items_std_neg)

        '''
        loss 计算
        '''
        # 主要的损失u+q-i 采样得到的uqi 计算重构误差
        loss_main, dis_pos, dis_neg = self.lossF_sigmod_ml(user_sample, query, product_sample, product_sample_neg)

        # 转移损失(KL损失) -->
        # 转移概率 loss current_mean, current_std, prior_mean, prior_std
        user_trans_loss = self.transfer_kl_loss(user_mean, user_std, user_mean_pri, user_std_pri, False, 'u')
        product_trans_pos_loss = self.transfer_kl_loss(item_mean_pos, item_std_pos, item_mean_pos_pri, item_std_pos_pri, False, 'i')
        product_trans_neg_loss = self.transfer_kl_loss(items_mean_neg, items_std_neg, items_mean_neg_pri, items_std_neg_pri, True, 'ni')
        
        # contrastive loss
        user_cpc_loss = self.loss_cpc(user_context_mean, user_context_std, user_mean, user_std, user_mean_neg, user_std_neg)
        item_cpc_loss = self.loss_cpc(item_context_mean, item_context_std, item_mean_pos, item_std_pos, items_mean_neg, items_std_neg)
        word_cpc_loss = self.loss_cpc(word_context_mean, word_context_std, pos_word_mean_emb, pos_word_std_emb, neg_word_mean_emb, neg_word_std_emb, query_len, is_word=True)
        recent_item_loss = self.loss_cpc(recent_items_context_mean, recent_items_context_std, user_mean, user_std, user_mean_neg, user_std_neg)
        recent_user_loss = self.loss_cpc(recent_users_context_mean, recent_users_context_std, item_mean_pos, item_std_pos, items_mean_neg, items_std_neg)

        loss = loss_main +\
            user_cpc_loss * torch.tensor(self.user_cpc_parameter).to(self.device)+\
            item_cpc_loss * torch.tensor(self.item_cpc_parameter).to(self.device)+\
            word_cpc_loss * torch.tensor(self.word_cpc_parameter).to(self.device)+\
            recent_item_loss * torch.tensor(self.recent_item_parameter).to(self.device)+\
            recent_user_loss * torch.tensor(self.recent_user_parameter).to(self.device)+\
            (user_trans_loss+product_trans_pos_loss+product_trans_neg_loss)*torch.tensor(self.kl_parameter).to(self.device)

        loss = (loss, loss_main, \
                user_trans_loss+product_trans_pos_loss+product_trans_neg_loss, user_cpc_loss, item_cpc_loss, word_cpc_loss, recent_item_loss, recent_user_loss)
        
        return loss, dis_pos, dis_neg
    
    def word_loss(self, itemOrUser, word_pos, word_len, word_neg):
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_review_len-int(i.item())) for i in word_len]).unsqueeze(2).to(self.device)
        word_pos = word_pos.mul(len_mask)
        itemOrUser.unsqueeze_(1)
        dis_pos = (itemOrUser - word_pos).norm(2, dim=2).mean(dim=1)
        dis_neg = (itemOrUser - word_neg).norm(2, dim=2).mean(dim=1)
        wl = torch.log(torch.sigmoid(dis_neg-dis_pos)).mean()*(-1.0)
        itemOrUser.squeeze_(1)
        return wl
        
    def reparameter(self, mean, std):

        std_z = torch.randn(std.shape, device=self.device)
        result = mean + torch.tensor(self.sigma_parameter).to(self.device)*std* Variable(std_z)  # Reparameterization trick

        return result
    
    
    def get_train_query_tanh_mean(self, query, query_len):
        '''
        input size: (batch, maxQueryLen)
        对query处理使用函数
        tanh(W*(mean(Q))+b)
        
        '''
        query = self.wordEmbedding_mean(query) # size: ((batch, maxQueryLen))) ---> (batch, len(query[i]), embedding)
        # query len mask 使得padding的向量为0
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_query_len-int(i.item())) for i in query_len]).unsqueeze(2).to(self.device)
        query = query.mul(len_mask)

        query = query.sum(dim=1).div(query_len.unsqueeze(1).float())
        query = self.queryLinear(query).tanh()

        return query

    def get_train_product_tanh_mean(self, products, product_len, max_len):
        '''
        get the mean of the product list
        '''
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(max_len-int(i.item())) for i in product_len]).unsqueeze(2).to(self.device)
        products = products.mul(len_mask)

        products = products.sum(dim=1).div(product_len.unsqueeze(1).float())
        
        # whether add the tanh one

        return products

    def transfer_mlp(self, prior, aim='u'):
        transfer_linear = getattr(self, 'transfer_linear_'+aim)
        current_hidden = transfer_linear(prior)
        transfer_mean = getattr(self, 'transfer_mean_'+aim)
        transfer_std = getattr(self, 'transfer_std_'+aim)
        return transfer_mean(current_hidden), transfer_std(current_hidden).mul(0.5).exp()

    
    def transfer_kl_loss(self, current_mean, current_std, prior_mean, prior_std, dim3=False, aim='u'):
        dim2 = current_mean.shape[1]
        if (dim3 == False):
            current_transfer_mean = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)
            current_transfer_std = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)
            for i in range(self.sample_num):
                prior_instance = self.reparameter(prior_mean, prior_std)
                cur_instance = self.transfer_mlp(prior_instance, aim)
                current_transfer_mean += cur_instance[0]
                current_transfer_std += cur_instance[1]

            # 取多个采样的Q(Zt-1)分布的均值为最终的loss 计算使用的P(Zt|B1:t-1)分布
            current_transfer_mean = current_transfer_mean.div(self.sample_num)
            current_transfer_std = current_transfer_std.div(self.sample_num**2)

            kl_loss = self.DKL(current_mean, current_std, current_transfer_mean, current_transfer_std)
        else:
            current_transfer_mean = torch.zeros((self.batch_size, dim2, self.embedding_dim), device=self.device)
            current_transfer_std = torch.zeros((self.batch_size, dim2, self.embedding_dim), device=self.device)
            for i in range(self.sample_num):
                prior_instance = self.reparameter(prior_mean, prior_std)
                cur_instance = self.transfer_mlp(prior_instance, aim)
                current_transfer_mean += cur_instance[0]
                current_transfer_std += cur_instance[1]

            # 取多个采样的Q(Zt-1)分布的均值为最终的loss 计算使用的P(Zt|B1:t-1)分布
            current_transfer_mean = current_transfer_mean.div(self.sample_num)
            current_transfer_std = current_transfer_std.div(self.sample_num)

            kl_loss = self.DKL(current_mean, current_std, current_transfer_mean, current_transfer_std, True)
        
        return kl_loss
    
    
    '''
    KL 误差
    KL(Q(Zt)||P(Zt|B1:t-1))
    P(Zt|B1:t-1) 使用采样计算～～1/K sum_{i=1}^K(P(Zt|Z_{i}t-1))
    '''
    def DKL(self, mean1, std1, mean2, std2, neg = False):
        var1 = std1.pow(2) + self.esp
        var2 = std2.pow(2) + self.esp
        mean_pow2 = (mean2-mean1)*(torch.tensor(1.0, device=self.device)/var2)*(mean2-mean1)
        tr_std_mul = (torch.tensor(1.0,  device=self.device)/var2)*var1
        if (neg == False):
            dkl = (torch.log(var2/var1)-1+tr_std_mul+mean_pow2).mul(0.5).sum(dim=1).mean()
        else:
            dkl = (torch.log(var2/var1)-1+tr_std_mul+mean_pow2).mul(0.5).sum(dim=2).sum(dim=1).mean()
        return dkl
    
    def word_DKL(self, mean1, std1, mean2, std2, query_len, neg = False):
        var1 = std1.pow(2) + self.esp
        var2 = std2.pow(2) + self.esp
        mean_pow2 = (mean2-mean1)*(torch.tensor(1.0, device=self.device)/var2)*(mean2-mean1)
        tr_std_mul = (torch.tensor(1.0,  device=self.device)/var2)*var1
        dkl = (torch.log(var2/var1)-1+tr_std_mul+mean_pow2)
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_query_len-int(i.item())) for i in query_len]).unsqueeze(2).to(self.device)

        if (neg == False):

            dkl = dkl.mul(len_mask)
            dkl = dkl.mul(0.5).sum(dim=2).mean()
        else:
            dkl = dkl.mul(len_mask.unsqueeze(3))
            
            dkl = dkl.mul(0.5).sum(dim=3).sum(dim=2).mean()

        return dkl
    
    '''
    主损失 重构误差
    -Eq(log{P(Bt|Zt)})
    '''
    def lossF_sigmod_ml(self, user, query, item_pos, items_neg):
        u_plus_q = user+query
        dis_pos = (u_plus_q - item_pos).norm(2, dim=1).mul(5.)
        u_plus_q.unsqueeze_(1)
        dis_neg = (u_plus_q - items_neg)
        dis_neg = dis_neg.norm(2,dim=2)
        dis_pos = dis_pos.view(-1,1)

        batch_loss = torch.log(torch.sigmoid(dis_neg-dis_pos)).sum(dim=1)*(-1.0)
        return batch_loss.mean() , dis_pos.mean(), dis_neg.mean()
    
    def loss_cpc(self, entity_mean, entity_std, pos_entity_mean, pos_entity_std, neg_entity_mean, neg_entity_std, query_len=None, is_word=False):
        '''
        compute the contrastive loss of entity
        '''
        if is_word == False:
            kl_pos = self.DKL(entity_mean, entity_std, pos_entity_mean, pos_entity_std, False)
            kl_neg = self.DKL(entity_mean.unsqueeze(1), entity_std.unsqueeze(1), neg_entity_mean, neg_entity_std, True)

        else:
            result = 0
            
            kl_pos = self.word_DKL(entity_mean, entity_std, pos_entity_mean, pos_entity_std, query_len, False)
            kl_neg = self.word_DKL(entity_mean.unsqueeze(2), entity_std.unsqueeze(2), neg_entity_mean.unsqueeze(1), neg_entity_std.unsqueeze(1), query_len, True)

        return kl_pos.mul(5.)+kl_neg


    # def loss_cpc(self, entity, pos_entity, neg_entity):
    #     '''
    #     compute the loss of entity
    #     '''
    #     dis_pos = (entity-pos_entity).norm(2, dim=1).mul(5.)
    #     dis_pos = dis_pos.view(-1,1)

    #     entity.unsqueeze_(1)
    #     dis_neg = (entity - neg_entity).norm(2, dim=2)

    #     batch_loss = torch.log(torch.sigmoid(dis_neg-dis_pos)).sum(dim=1)*(-1.0)
        
    #     return batch_loss.mean()

    def loss_contrastive_product(self, user, item, neg_item):
        '''
        calculate the distance between the user and the private interset or the public interest
        '''

        dis_pos = (user - item).norm(2, dim=1).mul(5.)
        dis_pos = dis_pos.view(-1,1)

        user.unsqueeze_(1)
        dis_neg = (user - neg_item)
        dis_neg = dis_neg.norm(2,dim=2)
        dis_pos = dis_pos.view(-1,1)
        batch_loss = torch.log(torch.sigmoid(dis_neg-dis_pos)).sum(dim=1)*(-1.0)
        
        return batch_loss.mean()