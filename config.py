# choose dataset name
dataset = 'Toys_and_Games'

# paths
main_review_path = '/home/binw/data/Amazon/review/Part/'
main_meta_path = '/home/binw/data/Amazon/metadata/'

stop_file = './stopwords.txt'

processed_path = './processed/'
out_path = './out/'
#result_path = 'result.csv'

# hyper parameter
'''
实验参数
'''
epoch = 500
embedding_dim = 50
out_size = 10
batch_size = 256
neg_sample_num = 0
dataLen = 0
batch_num = 0
full_len = 0
time_bin_num = 0
total_epoch = 2
gpu_id = 6

random_seed = 3

# contrastive learning parameter
cpc_para = 1e-2
dynamic_para = 1e-2