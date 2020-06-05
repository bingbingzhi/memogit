#-*-coding:utf-8-*-
import time
import bloscpack as bp
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV as rsc
from sklearn.model_selection import GridSearchCV as gsc
import scipy.sparse as sp
import numpy as np
from scipy.stats import expon
from extract_fr_tense import *


t = time.time()
X_train = bp.unpack_ndarray_from_file('../../data/fine_bert_train_X.blp')
#X_train = sp.load_npz('../data/X_train.npz')
idx_train = bp.unpack_ndarray_from_file('../../data/fine_bert_train_idx.blp')
#X_dev = bp.unpack_ndarray_from_file('../data/X_dev.blp')
#y_dev = bp.unpack_ndarray_from_file('../data/y_dev.blp')
X_test = bp.unpack_ndarray_from_file('../../data/fine_bert_test_X.blp')
#X_test = sp.load_npz('../data/X_test.npz')
idx_test = bp.unpack_ndarray_from_file('../../data/fine_bert_test_idx.blp')
#tsizeMB = sum(i.size*i.itemsize for i in (X_train,X_test))/2**20
def idx2label(idx_data):
	data=idx_data.tolist()
	y_list=[]
	for idx in data:
		y_list.append(sent2tense[idx])
	return y_list
y_train = idx2label(idx_train)
y_test = idx2label(idx_test)

t1 = time.time() - t
print("loading time = %.2f " % (t1))

#X_train = sp.csr_matrix(X_train)
#X_dev = sp.csr_matrix(X_dev)
#X_test = sp.csr_matrix(X_test)
#y_train=y_train.tolist()
#y_dev = y_dev.tolist()
#y_test= y_test.tolist()
#print(X_train.shape)
#print(len(y_train))
#model = LinearSVC(C=0.02,class_weight="balanced",random_state=42)

s1=time.perf_counter()
# train the model on train set
model = LinearSVC(C=0.02,class_weight="balanced",random_state=42)
#model = LinearSVC(C=0.02,random_state=42)
model.fit(X_train,y_train)
# print prediction results
y_predTrain1 = model.predict(X_train)
print(accuracy_score(y_train, y_predTrain1))
y_pred1 = model.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
e1 = time.perf_counter()
print ("################## training without tuning: ",e1-s1)

"""
s2=time.perf_counter()
# defining parameter range
params = {"C": [0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.1]}
#params = {"C": expon(scale=100)}

#model = LinearSVC(random_state=42)
tense_clf = gsc(model,params,n_jobs=5,refit = True)
#fitting the model for grid search
#tense_clf.fit(X_dev,y_dev)
tense_clf.fit(X_train,y_train)
# print best parameter after tuning
print(tense_clf.best_params_)
# print how our model looks after hyper-parameter tuning
print(tense_clf.best_estimator_)

#tense_clf.fit(X_train,y_train)

y_predTrain = tense_clf.predict(X_train)
print(accuracy_score(y_train, y_predTrain))
y_predTest = tense_clf.predict(X_test)
print(accuracy_score(y_test, y_predTest))
print(classification_report(y_test,y_predTest))
e2=time.perf_counter()
print ("################## training with tuning: ",e2-s2)
"""
