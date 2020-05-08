#-*-coding:utf-8-*-
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV as rsc
from sklearn.model_selection import GridSearchCV as gsc
from scipy.stats import expon
#from extract_features import *


t = time.time()
X_train = bp.unpack_ndarray_from_file('../data/X_train.blp')
y_train = bp.unpack_ndarray_from_file('../data/y_train.blp')
X_dev = bp.unpack_ndarray_from_file('../data/X_dev.blp')
y_dev = bp.unpack_ndarray_from_file('../data/y_dev.blp')
X_test = bp.unpack_ndarray_from_file('../data/X_test.blp')
X_test = bp.unpack_ndarray_from_file('../data/y_test.blp')
t1 = time.time() - t
print("loading time = %.2f (%.2f MB/s)" % (t1, tsizeMB/t1))

y_train=y_train.tolist()
y_dev = y_dev.tolist()
y_test= y_test.tolist()

s1=time.perf_counter()
# train the model on train set
model = LinearSVC(random_state=42)
model.fit(X_train,y_train)

# print prediction results
y_predTrain1 = model.predict(X_train)
print(accuracy_score(y_train, y_predTrain1))
y_pred1 = model.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
e1 = time.perf_counter()
print ("################## training without tuning: ",e1-s1)

s2=time.perf_counter()
# defining parameter range
params = {"C": [0.0001,0.001,0.02,0.03,0.04,0.05,0.06,0.1,1,10,100,1000,10000,100000]}
#params = {"C": expon(scale=100)}

#model = LinearSVC(random_state=42)
tense_clf = gsc(model,params,refit = True, cv=3,scoring='f1_macro')
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
print ("################## training with tuning: ",e1-s1)

"""
def plotSVC(title):
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() — 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() — 1, X[:, 1].max() + 1
  h = (x_max / x_min)/100
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
 plt.subplot(1, 1, 1)
 Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
 Z = Z.reshape(xx.shape)
 plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
 plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
 plt.xlabel(‘Sepal length’)
 plt.ylabel(‘Sepal width’)
 plt.xlim(xx.min(), xx.max())
 plt.title(title)
 plt.show()
"""