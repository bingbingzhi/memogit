#-*-coding:utf-8-*-
import pandas as pd
from sklearn.model_selection import train_test_split

df_data = pd.read_csv('../data/tensePred_data.csv')
df_bert = pd.DataFrame{'guid':df_data['Index'],
						'label':df_data['tense'],
						'text_fr':df_data['fr'],
						'text_zh':df_data['zh']}
# split into tr_dev,test
df_bert_train,df_bert_testdev = train_test_split(df_bert,test_size=0.2)
df_bert_dev,df_test = train_test_split(df_bert_testdev,test_size=0.5)

# create new data frame for test data
df_bert_test = pd.DataFrame({'guid':df_test['Index'],'text_zh':df_test['zh']}

# output tsv file, no header for train and dev
df_bert_train.to_csv('/home/bli/M2memo/bert-master/dataset/train.tsv',sep='\t',index=False,header=False)
df_bert_dev.to_csv('/home/bli/M2memo/bert-master/dataset/dev.tsv',sep='\t',index=False,header=False)
df_bert_test.to_csv('/home/bli/M2memo/bert-master/dataset/test.tsv',sep='\t',index=False,header=True)

