#-*-coding:utf-8-*-
import time
import bloscpack as bp
import scipy.sparse as sp
from extract_fr_tense import *

import numpy as np
from sklearn.manifold import TSNE
#import pandas as pd
#import seaborn as sns
import os
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# Importing sklearn and TSNE.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state we define this random state to use this value in TSNE which is a randmized algo.
RS = 25111993

# Importing matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
#%matplotlib inline

# Importing seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

t = time.time()
#X_train = bp.unpack_ndarray_from_file('../../data/fine_bert_train_X.blp')
#X_train = sp.load_npz('../data/X_train.npz')
#idx_train = bp.unpack_ndarray_from_file('../../data/fine_bert_train_idx.blp')
#X_dev = bp.unpack_ndarray_from_file('../data/X_dev.blp')
#y_dev = bp.unpack_ndarray_from_file('../data/y_dev.blp')
X_test = bp.unpack_ndarray_from_file('../../data/fine_bert_test_X.blp')
print(X_test.shape)
#X_test = sp.load_npz('../data/X_test.npz')
idx_test = bp.unpack_ndarray_from_file('../../data/fine_bert_test_idx.blp')
#tsizeMB = sum(i.size*i.itemsize for i in (X_train,X_test))/2**20
def idx2label(idx_data):
	data=idx_data.tolist()
	y_list=[]
	for idx in data:
		y_list.append(sent2tense[idx])
	return y_list
#y_train = idx2label(idx_train)
y_test = idx2label(idx_test)

t1 = time.time() - t
print("loading time = %.2f " % (t1))

# Loading the vector
#Data_1 = np.genfromtxt ('c:/Users/Imart/Desktop/text_vectors/excavator_text_vectors.csv', delimiter=",")
# Here we are importing KMeans for clustering Product Vectors
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_test)
# We can extract labels from k-cluster solution and store is to a list or a vector as per our requirement
Y=kmeans.labels_ # a vector

z = pd.DataFrame(Y.tolist()) # a list
PMCAT_List = ['Past','Pres','Fut']
# Fit the model using t-SNE randomized algorithm
digits_proj = TSNE(random_state=RS).fit_transform(X_test)

# An user defined function to create scatter plot of vectors
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 3))

    # We create a scatter plot.
    f = plt.figure(figsize=(48, 48))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=100,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each cluster.
    txts = []
    for i in range(3):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(PMCAT_List(i)), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

#print(list(range(0,3)))
sns.palplot(np.array(sns.color_palette("hls", 3)))
scatter(digits_proj, y_test)
plt.savefig('digits_tsne-generated_3_cluster.png', dpi=120)

