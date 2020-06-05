#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
from bert_serving.client import BertClient
import time
import bloscpack as bp
#import pkuseg

t=time.time()
bc = BertClient()

df_test = pd.read_csv('../data/test.tsv', delimiter='\t',header=0,names=['sentID','text_zh'])
#df_test = pd.read_csv('../data/test.tsv', delimiter='\t',usecols=[0,3],header=None)
#df_test.columns=['sentID','text_zh']
test_X_list = df_test['text_zh'].values.tolist()
test_idx = np.array(df_test['sentID'])
#print(test_X_list[:2])
#print(type(dev_X_list[:2]))

list_vec = bc.encode(test_X_list)
print(list_vec.shape)
#print(test_idx.shape)
bp.pack_ndarray_to_file(list_vec, '../data/fine_bert_test_X.blp')
bp.pack_ndarray_to_file(test_idx, '../data/fine_bert_test_idx.blp')
t1=time.time() - t
print("#conversion time: ",t1)
