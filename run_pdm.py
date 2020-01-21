# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from data_utils import *
from pdm_data_generation.pdmdb_generator import *

from pdm.pdm_train  import *
from pdm.pdm_monitoring import *

# %%

# Set path variables
ofile='./pdm/output_files/data/%s'
ifile='./pdm/input_data/%s'
model_path='./pdm/model/%s'
model_name='model.h5'
config_name='config.dictionary'

# %%
##load model and parameter
# from keras.models import load_model

model = load_model(model_path% model_name)
with open(model_path% config_name, 'rb') as config_file:
    param = pickle.load(config_file)

# %%

sequences = loadobj(ofile% "sequences.pick")
disturbed_sequences = loadobj(ofile% "disturbed_sequences.pick")

print("======== SEQUENCES =======")
seq_df=pd.DataFrame({'sequence':[], 'label':[]})
seq_df.sequence=pd.Series(disturbed_sequences[:])
# nbseq=len(seq_df.sequence)

seq_df.label=pd.Series(len(disturbed_sequences[:])*[1])
# print(seq_df.shape)
# print(seq_df.head(10))

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
save_text_file(result_normal, ofile% "result.txt")


# %%

# EXPERIMENT FOR DISTURBED SEQUENCES

seq_df.sequence=pd.Series(disturbed_sequences[:])

seq_df.label=pd.Series(len(disturbed_sequences[:])*[1])

result_disturbed=LSTMpred(list(seq_df.sequence))
# print(result)
save_text_file(result_disturbed, ofile% "result_disturbed.txt")


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

ytest = []
ytruth = []

for results in result_disturbed:
    ytest.extend([value[0] for value in results[0]])
    ytruth.extend(results[1])

# ytest = MultiLabelBinarizer().fit_transform(ytest)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ytest, ytruth)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ytest, ytruth, average='weighted')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, ytruth, average='weighted')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, ytruth, average='weighted')
print('F1 score: %f' % f1)
 
# # kappa
# kappa = cohen_kappa_score(ytest, yhat_classes)
# print('Cohens kappa: %f' % kappa)
# # ROC AUC
# auc = roc_auc_score(ytest, yhat_probs)
# print('ROC AUC: %f' % auc)

# confusion matrix
matrix = confusion_matrix(ytest, ytruth)
print(matrix)




# %%

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(ytruth, ytest)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))



# QUEDASTE ACA.
# ES NECESARIO EVALUAR LA ESTRUCTURA DEL DATASET DE ENTRADA, PARA LO CUAL TENES AUE GRFICAR LOS DATOS
# EN BASE A LOS GRAFICOS VER SI ES NECESARIO RE HACER EL GENERADOR DE SECUENCIAS
# UNA VEZ COMPLETADO LA GENERACION ES NECESARIO RE ENTRENAR Y LOGRAR UN BUEN RESULTADO CON EVENTOS Y TIEMPOS
# para el training considerar el uso de validation set propio para garantizar balance apropiado en los datos, evitar el uso del split del metofo fit
# I?PORTANTE VER LAS ?ETRICAS LUEGO DEL TESTING: PRECISION. RECALL. F1 (VER SI ES POSIBLE INTEGRARLAS EN EL TRAINING PARA GRAFICAR)
# HAY QUE GENERAR UNA FORMA DE AUTOMATIZAR LOS EXPERIMENTOS (GENERADOR AUTOMATICO DE CARPETAS)
# LUEGO. PROBAR CON LA CAPA CONVULCIONAL CONVLSTM





# %%
