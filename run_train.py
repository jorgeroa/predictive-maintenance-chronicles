# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from data_utils import *
from pdm_data_generation.pdmdb_generator import *
from pdm.pdm_train import *
from pdm.pdm_monitoring import *

from keras.models import load_model

from matplotlib import pyplot as plt


# %%

# # Load sequences generated with the PdM generator
sequences = loadobj(f_current_seq_normal_bin)
disturbed_sequences = loadobj(f_current_seq_disturbed_bin)

# %%

X = sequences[:]    # Copy sequences to X
train_size = int(len(X) * 0.7)  # 70% for training
X_train, X_test = X[0:train_size], X[train_size:len(X)]

print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(X_train)))
print('Testing Observations: %d' % (len(X_test)))


# %%

# Start training
print("####### Start traning ########")

h=train(X_train)
saveobj(f_history,h)

print("####### End traning ########")


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
seq_df.sequence=pd.Series(X_test[:])
# nbseq=len(seq_df.sequence)

seq_df.label=pd.Series(len(X_test[:])*[1])

# %%

# EXPERIMENT FOR NORMAL SEQUENCES
def LSTMpred(seqs):
    start_time = time.perf_counter()
    SS=[anomalydetect([seq],param,model) for seq in seqs]
    end_time=time.perf_counter()
    le=end_time-start_time
    return SS 

result_normal=LSTMpred(list(seq_df.sequence))
# print(result)
save_text_file(result_normal, fold_current_input_data+"/result.txt")

# %%

# Calculate metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MultiLabelBinarizer

y_ev_test = []
y_ev_truth = []
y_t_test = []
y_t_truth = []
values = []

for results in result_normal:
    y_ev_test.extend([value[0] for value in results[0]])
    y_ev_truth.extend(results[1])
    y_t_test.extend([value[0] for value in results[2]])
    y_t_truth.extend(results[3])

# y_ev_test = MultiLabelBinarizer().fit_transform(y_ev_test)

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
 
# # kappa
# kappa = cohen_kappa_score(y_ev_test, yhat_classes)
# print('Cohens kappa: %f' % kappa)
# # ROC AUC
# auc = roc_auc_score(y_ev_test, yhat_probs)
# print('ROC AUC: %f' % auc)

# confusion matrix
matrix = confusion_matrix(y_ev_test, y_ev_truth)
print(matrix)

# ########################################

# y_t_test = np.array(y_t_test)*60
# y_t_truth = np.array(y_t_truth)*60

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_t_test, y_t_truth)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_t_test, y_t_truth, average='macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_t_test, y_t_truth, average='macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_t_test, y_t_truth, average='macro')
print('F1 score: %f' % f1)
 
# # kappa
# kappa = cohen_kappa_score(y_t_test, yhat_classes)
# print('Cohens kappa: %f' % kappa)
# # ROC AUC
# auc = roc_auc_score(y_t_test, yhat_probs)
# print('ROC AUC: %f' % auc)

# confusion matrix
matrix = confusion_matrix(y_t_test, y_t_truth)
matrix
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
