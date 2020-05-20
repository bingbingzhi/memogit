#-*-coding:utf-8-*-
import time
import bloscpack as bp
from sklearn.svm import LinearSVC
import sklearn.metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV as rsc
from sklearn.model_selection import GridSearchCV as gsc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from scipy.stats import expon
import pandas as pd
import seaborn as sns
#from extract_features import *

np.set_printoptions(precision=2)
t = time.time()
#X_train = bp.unpack_ndarray_from_file('../data/X_train.blp')
X1_train = sp.load_npz('../data/p2_X_train.npz')
y1_train = bp.unpack_ndarray_from_file('../data/p2_y_train.blp')
X2_train = sp.load_npz('../data/p2_no_pres_X_train.npz')
y2_train = bp.unpack_ndarray_from_file('../data/p2_no_pres_y_train.blp')
#X_test = bp.unpack_ndarray_from_file('../data/X_test.blp')
X_test = sp.load_npz('../data/p2_X_test.npz')
y_test = bp.unpack_ndarray_from_file('../data/p2_y_test.blp')
#tsizeMB = sum(i.size*i.itemsize for i in (X_train,X_test))/2**20
t1 = time.time() - t
print("loading time = %.2f " % (t1))
#print("X1_train shape: ",X1_train.shape)
#print("X2_train shape: ",X2_train.shape)
#X_train = sp.csr_matrix(X_train)
#X_dev = sp.csr_matrix(X_dev)
#X_test = sp.csr_matrix(X_test)
y1_train=y1_train.tolist()
y2_train=y2_train.tolist()
#print(y1_train[:30])
#print(y2_train[:30])
#y_dev = y_dev.tolist()
y_test= y_test.tolist()
#print(y_test[:100])
s1=time.perf_counter()


# modèle pres non-présent
# train the model on train set
model1 = LinearSVC(C=0.01,class_weight="balanced")
model2 = LinearSVC(C=0.01)#,class_weight="balanced",random_state=42)

params = {"C": [0.005,0.006,0.007,0.008,0.009,0.01,0.014,0.015,0.016,0.017,0.02]}
pres_notPres_clf = gsc(model1,params,n_jobs=5,refit = True)
# fitting the model for grid search
pres_notPres_clf.fit(X1_train,y1_train)
# print best parameter after tuning
print(pres_notPres_clf.best_params_)
# print how our model looks after hyper-parameter tuning
print(pres_notPres_clf.best_estimator_)

y1_predTrain = pres_notPres_clf.predict(X1_train)
print(accuracy_score(y1_train, y1_predTrain))
y1predTest = pres_notPres_clf.predict(X_test)
print(y1predTest.shape)
print(y1predTest[:30])


no_pres_idxs= np.where(y1predTest == 3)[0]
X_test=X_test.todense()
n_X_test=X_test.take(no_pres_idxs,axis=0)
n_X_test = sp.csr_matrix(n_X_test)


past_fut_clf = gsc(model2,params,n_jobs=5,refit = True)
past_fut_clf.fit(X2_train,y2_train)
print(past_fut_clf.best_params_)
# print how our model looks after hyper-parameter tuning
print(past_fut_clf.best_estimator_)
#labels = ['Pres','NoPres']
# print prediction results
y2_predTrain = past_fut_clf.predict(X2_train)
print(accuracy_score(y2_train, y2_predTrain))
#print(confusion_matrix(y_train,y_predTrain1,labels = [0,1,2,3]))
y2predTest = past_fut_clf.predict(n_X_test)
c=0
for i in no_pres_idxs:
	y1predTest[i] = y2predTest[c]
	c+=1

print(y1predTest[:30])
print(accuracy_score(y_test,y1predTest))
print(classification_report(y_test,y1predTest))
cm = confusion_matrix(y_test,y1predTest)




#Plot with confusion matrix with scikit-learn without a classifier/estimator
def plot_confusion_matrix(y_test, y1predTest, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            #title = 'Normalized confusion matrix'
            title = 'Accuracy = {0:.2f}'.format(accuracy_score(y_test, y1predTest))
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y1predTest)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_test, y1predTest)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y1predTest, classes=['Past','Fut','Pres'], normalize=True,
                      title=None)
plt.savefig('2binary.png',dpi=200)
