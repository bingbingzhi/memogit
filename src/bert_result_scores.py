#-*-coding:utf-8-*-
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import bloscpack as bp

y_test =bp.unpack_ndarray_from_file('/home/bli/M2memo/memogit/data/bert_test_y.blp')
#read the original test data for the text and id
df_test = pd.read_csv('dataset/test.tsv', sep='\t')

#read the results data for the probabilities
df_result = pd.read_csv('bert_output/test_results.tsv', sep='\t', header=None)
#create a new dataframe
df_map_result = pd.DataFrame({'guid': df_test['guid'],
    'text': df_test['text'],
    'label': df_result.idxmax(axis=1)})
#view sample rows of the newly created dataframe
df_map_result.sample(10)

y1predTest = np.array(df_map_result['label'])

print(classification_report(y_test,y1predTest))


cm = confusion_matrix(y_test,y1predTest)
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
#plot_confusion_matrix(y_test, y1predTest, classes=['Past-PC','Pres','Fut'], normalize=True,
#                      title=None)
#plt.savefig('bert_classifier.png',dpi=200)