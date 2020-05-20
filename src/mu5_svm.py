#-*-coding:utf-8-*-
import time
import bloscpack as bp
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV as rsc
from sklearn.model_selection import GridSearchCV as gsc
import scipy.sparse as sp

from scipy.stats import expon
#from extract_features import *


t = time.time()
#X_train = bp.unpack_ndarray_from_file('../data/X_train.blp')
X_train = sp.load_npz('../data/mu5_X_train.npz')
y_train = bp.unpack_ndarray_from_file('../data/mu5_y_train.blp')
#X_dev = bp.unpack_ndarray_from_file('../data/X_dev.blp')
#y_dev = bp.unpack_ndarray_from_file('../data/y_dev.blp')
#X_test = bp.unpack_ndarray_from_file('../data/X_test.blp')
X_test = sp.load_npz('../data/mu5_X_test.npz')
y_test = bp.unpack_ndarray_from_file('../data/mu5_y_test.blp')
#tsizeMB = sum(i.size*i.itemsize for i in (X_train,X_test))/2**20
t1 = time.time() - t
print("loading time = %.2f " % (t1))

#X_train = sp.csr_matrix(X_train)
#X_dev = sp.csr_matrix(X_dev)
#X_test = sp.csr_matrix(X_test)
y_train=y_train.tolist()
#y_dev = y_dev.tolist()
y_test= y_test.tolist()

s2=time.perf_counter()
model = LinearSVC(C=0.02,class_weight="balanced",random_state=42)
#model = LinearSVC(random_state=42)
# defining parameter range
params = {"C": [0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.1]}

tense_clf = gsc(model,params,n_jobs=5,refit = True)

#fitting the model for grid search
tense_clf.fit(X_train,y_train)
# print best parameter after tuning
print(tense_clf.best_params_)
# print how our model looks after hyper-parameter tuning
print(tense_clf.best_estimator_)


y_predTrain = tense_clf.predict(X_train)
print(accuracy_score(y_train, y_predTrain))
y_predTest = tense_clf.predict(X_test)
print(accuracy_score(y_test, y_predTest))
print(classification_report(y_test,y_predTest))
e2=time.perf_counter()
print ("################## training with tuning: ",e2-s2)


