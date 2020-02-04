# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from data_utils import *
from pdm_data_generation.pdmdb_generator import *
from pdm.pdm_train import *
from pdm.pdm_monitoring import *

from keras.models import load_model

from matplotlib import pyplot as plt
import numpy as np

print("Experiment:",iexp)
print("Input Sequence:",iseq)
print("Model:",imodel)
print("Folder experiments:",fold_gen)
print("Folder sequences:",fold_seq)


# %%

# # Load sequences generated with the PdM generator
# sequences = loadobj(f_current_seq_normal_bin)
# disturbed_sequences = loadobj(f_current_seq_disturbed_bin)
sequences = loadobj(f_seq_normal_bin)
noisy_sequences = loadobj(f_seq_noisy_bin)
disturbed_sequences = loadobj(f_seq_disturbed_bin)

# %%

X = sequences[:]    # Copy sequences to X
train_size = int(len(X) * 0.7)  # 70% for training
X_train, X_test = X[0:train_size], X[train_size:len(X)]

print('Observations: %d' % (len(X_train)+len(X_test)+len(noisy_sequences)+len(disturbed_sequences)))
print('Training Observations: %d' % (len(X_train)))
print('Testing Observations: %d' % (len(X_test)))
print('Noisy Observations: %d' % (len(noisy_sequences)))
print('Disturbed Observations: %d' % (len(disturbed_sequences)))


# %%

# Start training
print("####### Start traning ########")

h=train(X_train)
saveobj(f_history,h)

print("####### End tranin+g ########")


# %%
##load model and parameter
# saveobj(f_history,h)

# model = load_model(f_current_model)
# param = loadobj(f_current_config)
model = load_model(f_model)
param = loadobj(f_config)

# %%
# TODO: After the training there should be a testing process here:
# It's important to check precision, recall, and F1

print("======== SEQUENCES =======")
seq_df=pd.DataFrame({'sequence':[], 'label':[]})
# seq_df.sequence=pd.Series(X_train[:]+X_test[:]+noisy_sequences[:]+disturbed_sequences[:])
seq_df.sequence=pd.Series(X_test[:]+noisy_sequences[:]+disturbed_sequences[:])
# nbseq=len(seq_df.sequence)

# seq_df.label=pd.Series(len(X_train[:])*[1]+len(X_test[:])*[1]+len(noisy_sequences[:])*[1]+len(disturbed_sequences[:])*[0])
seq_df.label=pd.Series(len(X_test[:])*[1]+len(noisy_sequences[:])*[1]+len(disturbed_sequences[:])*[0])

# %%

# EXPERIMENT FOR NORMAL SEQUENCES
def LSTMpred(seqs):
    start_time = time.perf_counter()
    # SS=[anomalydetect([seq],param,model) for seq in seqs]
    scores = []
    for seq in seqs:
        r = anomalydetect([seq],param,model)
        scores.append(r)

    end_time=time.perf_counter()
    le=end_time-start_time
    return scores 

result_normal=LSTMpred(list(seq_df.sequence))
# print(result)
# save_text_file(result_normal, fold_current_input_data+"/result.txt")
save_text_file(result_normal, fold_output_data+"/result.txt")

# %%
def decision(l,th=0.9):
    return [int(i>=th) for i in l]

def learn_threshold(predict,label):
    ss=[i for i in np.arange(1, .025, -0.025)]
    fl=[]
    f=0
    threshold=0
    for s in ss:
        y_pred=decision(predict,s)
        f1=metrics.f1_score(label, y_pred, average='macro')
        fl.append(f1)
        if(f<f1):
            f=f1
            threshold=s
    return ss,fl,f,threshold

def plot_threshold(ss,fl,f,threshold):
    plt.plot(ss, fl, label='F1-score') #blue
    # plt.plot(ss, fl) #blue
    plt.plot([threshold], [f],'rx',markersize=6) #red
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show() 

def plot_ROC(test_labels, test_predictions):
    fpr, tpr, thresholds = metrics.roc_curve(
        test_labels, test_predictions, pos_label=1)
    auc = "%.2f" % metrics.auc(fpr, tpr)
    title = 'ROC Curve, AUC = '+str(auc)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#000099", label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
    return fig

def savescore(filename,p,r,f):
    with open(filename, 'a+') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #spamwriter.writerow(["CaseID", "nb ch","nbseq","per", "accuracy", "recall", "f1 score", "execution time","memory usage"])
        spamwriter.writerow(['*',p,r,f])


# %%
labels = list(seq_df.label)
ss,fl,f,threshold=learn_threshold(result_normal,labels)
print("Threshold: ",threshold, " F1-score: ", f)
plot_threshold(ss,fl,f,threshold)

plot_ROC(seq_df.label,result_normal)

# %%

# Calculate metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.preprocessing import MultiLabelBinarizer

# y_ev_test = MultiLabelBinarizer().fit_transform(result_normal)
y_ev_test = decision(result_normal,threshold)
y_ev_truth = seq_df.label

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_ev_test, y_ev_truth)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_ev_test, y_ev_truth, average='macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_ev_test, y_ev_truth, average='macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_ev_test, y_ev_truth, average='macro')
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(y_ev_test, y_ev_truth)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_ev_test, y_ev_truth)
print('ROC AUC: %f' % auc)

# confusion matrix
matrix = confusion_matrix(y_ev_test, y_ev_truth)
# print(matrix)
# tn, fp, fn, tp = confusion_matrix(y_ev_test, y_ev_truth).ravel()
# print("tn:"+str(tn)+", fp:"+str(fp)+", fn:"+str(fn)+", tp:"+str(tp))
# print(tn, fp, fn, tp)
target_names = ['Anomaly', 'No anomaly']
print(classification_report(y_ev_test, y_ev_truth, target_names=target_names))

# %%

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm = np.array(matrix), 
                      normalize    = False,
                      target_names = target_names,
                      title        = "Confusion Matrix")


# %%

# If the model was just trained jump to the next cell. There is no need to reload the history from a file.
# Load history if the model is saved in a file.
history=loadobj(f_history)

# %%
print("*********************************")
print("*********************************")
print("*********************************")

plt.plot(history.history['act_output_acc'])
plt.plot(history.history['val_act_output_acc'])
plt.title('Model accuracy for events')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()
# %%

plt.plot(history.history['time_output_acc'])
plt.plot(history.history['val_time_output_acc'])
plt.title('Model accuracy for time')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# %%

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%

plt.plot(history.history['act_output_loss'])
plt.plot(history.history['val_act_output_loss'])
plt.title('Model loss for events')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# %%

plt.plot(history.history['time_output_loss'])
plt.plot(history.history['val_time_output_loss'])
plt.title('Model loss for time')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# %%

plt.plot(history.history['time_output_mean_squared_error'])
plt.plot(history.history['val_time_output_mean_squared_error'])
plt.title('Model mse for time')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
