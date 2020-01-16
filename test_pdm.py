# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from pdm_utils import *
from pdmdb_generator import *

from pdm.pdm_train  import *
from pdm.pdm_monitoring import *

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from memory_profiler import memory_usage

from  IPython import display
from matplotlib import pyplot as plt

import seaborn as sns

# %% [markdown]
# ############ Generate Data #####################
# ################################################

# %%

# Instantiate the generator
generator=pdmdb_generator(nbitems=10,l=15, lp=7, dc=1, minstart=0.1, minduration=1, maxduration=1440)

# Stores all the chronicle sequences generated by the generator 
chro_sequences = generator.generate(nb=300, l=5, npat=3, th=0.5)

# Stores all the sequences taken from each chronicle sequence of the variable chro_sequences
sequences=[s.seq for s in chro_sequences]

# The disturbed chronicles to be used for generating disturbed sequences
disturbed_chronicles = []

for c in generator.patterns:
    # Probability of: [1:removing items, 2:modifying constraints, 3:adding items]
    prob = [0,1,0]  # Set probability of modifying constraints to 1 and 0 for removing/adding
    # Generate disturbed a chronicle from chronicle c with probability prob
    disturbed_chronicle = generator.pattern_generator.generate_similar(c,prob)
    disturbed_chronicles.append(disturbed_chronicle)

# Stores all the disturbed chronicle sequences generated by the generator 
disturbed_chro_sequences = generator.generate(nb=300, l=5, npat=3, th=0.5, patterns=disturbed_chronicles)

# Stores all the sequences taken from each chronicle sequence of the variable disturbed_chro_sequences
disturbed_sequences=[s.seq for s in disturbed_chro_sequences]

# %%

# Removes from sequences and disturbed_sequences all events equal to -1
sequences=filterseq(sequences)
disturbed_sequences=filterseq(disturbed_sequences)
# %% [markdown]
# ############ Saves Data #####################
# ################################################

# %%

# Serialize sequences
serialization2(sequences,"sequences.txt")
serialization2(disturbed_sequences,"disturbed_sequences.txt")

# %%

# Serialize sequences with time
serialization(sequences,"sequencesT.txt")
serialization(disturbed_sequences,"disturbed_sequencesT.txt")

# %%

saveobj("sequences.pick",sequences)
saveobj("disturbed_sequences.pick",disturbed_sequences)
# %%

sequencesF=filterseq(sequences)
disturbed_sequencesF=filterseq(disturbed_sequences)
# Serialize sequences
serialization2(sequencesF,"sequences.txt")
serialization2(disturbed_sequencesF,"disturbed_sequences.txt")
# %%
sequences = loadobj("sequences.pick")
disturbed_sequences = loadobj("disturbed_sequences.pick")


# %%
# from pdm.pdm_train import *
print("####### Start of traning ########")

h=train(sequences)
saveobj("history.hist",h)

print("####### End of traning ########")

# %%
##load model and parameter
# from keras.models import load_model

model = load_model('./pdm/output_files/model.h5')
with open('./pdm/output_files/config.dictionary', 'rb') as config_file:
    param = pickle.load(config_file)
# %%
# summarize history for loss
h1=loadobj("history.hist")
history=h1

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

sequences = loadobj("sequences.pick")
disturbed_sequences = loadobj("disturbed_sequences.pick")

print("======== SEQUENCES =======")
seq_df=pd.DataFrame({'sequence':[], 'label':[]})
seq_df.sequence=pd.Series(disturbed_sequences[:])
# nbseq=len(seq_df.sequence)

seq_df.label=pd.Series(len(disturbed_sequences[:])*[0])
# print(seq_df.shape)
# print(seq_df.head(10))

# %%
import pdm.pdm_monitoring

def LSTMpred(seqs):
    start_time = time.perf_counter()
    SS=[anomalydetect([seq],param,model) for seq in [seqs[0]]]
    end_time=time.perf_counter()
    le=end_time-start_time
    return SS 

result=LSTMpred(list(seq_df.sequence))
# print(result)
save_text_file(result,"result.txt")

# %%
# ############### generate test data ######################
# nbitems=50 #nb vocab
# seqlen=15 #len ch
# ch=DB_c[:]
# nbc=len(ch)
# generator=chrodb_generator(nbitems=50,l=25, lp=15)
# DB_seq=generator.generate(nb=50, l=25, npat=len(ch), th=.5,patterns=ch,pert=-1)
# p_seq0=generator.generate(nb=20, l=25, npat=len(ch), th=.5,patterns=ch,pert=0)
# p_seq1=generator.generate(nb=20, l=25, npat=len(ch), th=.5,patterns=ch,pert=1)

# #p_seq0=filterseq([s.seq for s in p_seq0])
# #p_seq1=filterseq([s.seq for s in p_seq1])
# #DB_seq=filterseq([s.seq for s in DB_seq])
# DB_seq_label=len(DB_seq)*[1]
# #trainseq.extend(pseq)
# print(len(DB_seq))
# print(len(p_seq0))
# print(len(p_seq1))

# %% [markdown]
# ############### Data Frame #################
# ###########################################
# %%
nbitems=50 #nb vocab
seqlen=10 #len ch
# p_seq0=[]
# p_seq1=[]
DB_seq=trainseq[:]

DB_seq[:]+p_seq0[:]+p_seq1[:]

times = []
events=[]

for i,seq in enumerate(DB_seq): #the rows are "ChID,sequence,TC"
    if len(seq)==0:
        continue
    for t,e in seq:
        events.append(e)
        times.append(t)

    # lines.append(evnts)
    # timeseqs.append(times)

print(len(events))
print(len(times))
# seq_df=pd.DataFrame({'event':events, 'time':times})
seq_df=pd.DataFrame({'event':[], 'time':[]})
s=seq_df.event=pd.Series(events)
seq_df.time=pd.Series(times)

# seq_df.sequence=pd.Series(DB_seq[:]+p_seq0[:]+p_seq1[:])

s.value_counts()
# seq_df.plot(kind='bar')

# %%

print("======== SEQUENCES =======")
seq_df=pd.DataFrame({'sequence':[], 'label':[]})
seq_df.sequence=pd.Series(disturbed_sequences[:])
# nbseq=len(seq_df.sequence)

seq_df.label=pd.Series(len(disturbed_sequences[:])*[0])
# print(seq_df.shape)
# print(seq_df.head(10))

#for e in seq_df.sequence:
#    print(e)

# %%

dataset = seq_df.copy()

print("head:")
train_dataset = dataset.sample(frac=0.8,random_state=0)
train_dataset.head()
# train_dataset.index
train_dataset.index
# train_dataset.describe()
# train_dataset['sequence']
# train_dataset.iloc[3]

# train_stats = train_dataset.describe()
# train_stats.pop("sequence")
# train_stats = train_stats.transpose()
# train_stats


# %%

import seaborn as sns
sns.pairplot(train_dataset[["sequence", "label"]], diag_kind="kde")


# %%

print("tail:")
test_dataset = dataset.drop(train_dataset.index)
test_dataset.tail()

# %%
# ############### Training phase #################
# ################################################


# %%
# ############### Plot phase #################
# ################################################

# %%
print(h)
# plotter = tfdocs.plots.HistoryPlotter(metric = 'categorical_crossentropy', smoothing_std=10)
# plotter.plot(h)
# plt.ylim([0.5, 0.7])
# %%
%load_ext tensorboard
import tensorflow as tf 

print("TF VERSION: ",tf.__version__)

%tensorboard --logdir LSTM/output_files/summary

# %% [markdown]
# ############ LSTM ########################

# %%
##load model and parameter
model = load_model('./LSTM/output_files/model.h5')
with open('./LSTM/output_files/config.dictionary', 'rb') as config_file:
    param = pickle.load(config_file)
#seqs=read_text_file("./LSTM/input_data/testseqs.txt")
#trainseq=[ deserialization(line) for line in seqs ]
def LSTMpred(seqs):
    start_time = time.perf_counter()
    SS=[anomalydetect([seq],param,model) for seq in [filterseq(seqs)[0]]]
    end_time=time.perf_counter()
    le=end_time-start_time
    return SS,le
#SS,le= LSTMpred(trainseq)
#SS,le= LSTMpred(list(seq_df.sequence))
#lmu=np.mean(memory_usage((LSTMpred,(list(seq_df.sequence),))))

#print(SS)


# %%
j=0
nbc=1
pert=.3
for i in range(15,150,15):
    nbitems=np.mean([len(s) for s in list(seq_df.sequence)[j:i]])
    SS,le= LSTMpred(list(seq_df.sequence)[j:i])
    # JGT: lmu=np.mean(memory_usage((LSTMpred,(list(seq_df.sequence),))))
    print(SS)
    seuil=0.6
    #ss,fl,f,seuil=learn_threshold(SS,seq_df.label[j:i])
    y_pred=decision(SS,seuil)
    print(y_pred)

    #assert(len(seq_df)==len(y_pred))
    # Display the confusion matrix
    #print(metrics.confusion_matrix(seq_df.label[j:i], y_pred))
    # Calculate the classification rate of this classifier
    #p=metrics.accuracy_score(seq_df.label[j:i], y_pred)
    #r=metrics.recall_score(seq_df.label[j:i], y_pred, average='macro')
    #f1=metrics.f1_score(seq_df.label[j:i], y_pred, average='macro')
    #p1,r1,f,_=metrics.precision_recall_fscore_support(seq_df.label[j:i], y_pred, average='micro')
    #print(f1,r,p)
    # JGT: savescore("LSTM3.csv",p,r,f1,le,lmu)

    p1,r1,f,_=metrics.precision_recall_fscore_support([seq_df.label[j]], y_pred, average='micro')
    savescore("LSTM.csv",p1,r1,f,le,0)

    j=i


# %%
j=0
nbc=3
pert=.3
for s in trainseq[8:]:
    #s=filterseq([s])[0]
    nbitems=len(s)
    SS,le= LSTMpred([s])
    # lmu=np.mean(memory_usage((LSTMpred,([s],))))
    print(SS)
    ss,fl,f,seuil=learn_threshold(SS,[seq_df.label[j]])
    y_pred=decision(SS,seuil)
    print(y_pred)

    #assert(len(seq_df)==len(y_pred))
    # Display the confusion matrix
    print(metrics.confusion_matrix([seq_df.label[j]], y_pred))
    # Calculate the classification rate of this classifier
    #p=metrics.accuracy_score(seq_df.label[j:i], y_pred)
    #r=metrics.recall_score(seq_df.label[j:i], y_pred, average='macro')
    #f1=metrics.f1_score(seq_df.label[j:i], y_pred, average='macro')
    p1,r1,f,_=metrics.precision_recall_fscore_support([seq_df.label[j]], y_pred, average='micro')
    # print(f1,r,p)
    savescore("LSTM2.csv",p1,r1,f,le,0)
    j +=1


# %%



# %%
#seq_df.label=len(trainseq)*[1]
ss,fl,f,seuil=learn_threshold(SS,seq_df.label)
print(seuil,f)
plot_threshold(ss,fl,f,seuil)
y_pred=decision(SS,seuil)
print(y_pred)

# Calculate the classification rate of this classifier
lp=metrics.accuracy_score(seq_df.label, y_pred)
lr=metrics.recall_score(seq_df.label, y_pred, average='macro')
lf1=metrics.f1_score(seq_df.label, y_pred, average='macro')
print(lf1)


# %%
#nbseq=14
pert=.3
#nbc=5
lmu=np.mean(memory_usage((LSTMpred,(list(seq_df.sequence),))))
# Display the confusion matrix
print(metrics.confusion_matrix(seq_df.label, y_pred))
savescore("LSTM.csv",lp,lr,lf1,le,lmu)

# %% [markdown]
# ################### train LSTM model ####################

# %%
################### train LSTM model ####################
i=15
saveobj("trainseq.dictionary".format(i),trainseq)
trainseq=loadobj("trainseq.dictionary".format(i))


# %%
# summarize history for loss
history=h
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()








