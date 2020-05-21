#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
from bert_serving.client import BertClient
bc = BertClient(ip='194.254.200.26')
df_dev_X = pd.read_csv('../data/dev.tsv',usecols =[3], delimiter='\t',header=None)
df_dev_idx = pd.read_csv('../data/dev.tsv',usecols =[0], delimiter='\t',header=None)
dev_X = np.array(df_dev_X)
dev_X_list = dev_X.tolist()
dev_idx = np.array(df_dev_idx)
#print(dev_X_list[:2])

list_vec = bc.encode(dev_X_list[:2])
print(list_vec)