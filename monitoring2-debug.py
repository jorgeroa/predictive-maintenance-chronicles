# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from recognition.chronicle_recognition import Chronicle
from seq_generation.chronicle_generator import *
from monitoring2 import *
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from LSTM.LSTMtrainBin  import *
from LSTM.LSTMmonotoring import *
from memory_profiler import memory_usage

from  IPython import display
from matplotlib import pyplot as plt

import seaborn as sns



# %%

ofile='./LSTM/output_files/data/%s'
ifile='./LSTM/input_data/%s'

def filterseq(DBseq):
    """
    @JGT: Removes from DBseq all events equal to -1
    """
    seqs=[]
    for seq in DBseq:
        #seqs.append([(t,e) for (t,e) in seq if e in c.sequence])
        seqs.append([(t,e) for (t,e) in seq if e != -1])
    return [s for s in seqs if len(s)>1]

"UCAD.csv , LSTM.csv"
def savescore(filename,p,r,f,e,mu):
    with open(ofile % filename, 'a+') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #spamwriter.writerow(["CaseID", "nb ch","nbseq","per", "accuracy", "recall", "f1 score", "execution time","memory usage"])
        # JGT: spamwriter.writerow(['*',nbc,nbseq,pert,p,r,f,e,mu,nbitems,seqlen])
        spamwriter.writerow(['*',nbc,nbseq,pert,p,r,f,e,mu,nbitems])
        
import re 
def deserialization(input):  
     # \d+ is a regular expression which means 
     # one or more digit 
     # output will be like ['100','564','365'] 
    numbers = re.findall('\d+',input) 

     # now we need to convert each number into integer 
     # int(string) converts string into integer 
    numbers = list(map(int,numbers))
    l=[]
    for i in range(0,len(numbers)-1,2):
        l.append((numbers[i],numbers[i+1]))
    return l

def read_text_file(filename):
    print('Reading file ' + filename + "...")
    with open(filename, "r", encoding='utf8') as textfile:
        L = []
        for line in textfile:
            L.append(line.strip())
        print('File contains ', len(L), "lines.\n")
        return L

def serialization(seqs,filename):
     with open(ifile%filename, "w+") as fichier:
        for seq in seqs:
            fichier.write(str(seq).strip('[]')+'\n')
            
def serialization2(seqs,filename):
     with open(ifile%filename, "w+") as fichier:
        for seq in seqs:
            s=[i[1] for i in seq ]
            fichier.write(str(s).strip('[]')+'\n')
import pickle          
def saveobj(filename,obj):
    with open(ofile% filename, 'wb') as config_file:
        pickle.dump(obj, config_file)
        
def loadobj(filename):
    with open(ofile% filename, 'rb') as config_file:
        return pickle.load(config_file)

def makelabel(seqs):
    """
    @JGT: Assigns value 1 as the label for the input sequences seqs
    """
    seqs_label=[]
    for seq in seqs :
        if len(seq)==0:
            #DB_seq_label.append(0)
            seqs.remove(seq)
        if len(seq)==1:
            seqs.remove(seq)
        else:
            seqs_label.append(1)
    return seqs_label

def learn_threshold(predict,label):
    ss=[i for i in np.arange(1, .025, -0.025)]
    #ss=[1,0.95,.9,.85,.8,.75,.7,.65,.6,.55,.5,.45,.4,.35,.3,.25,.2]
    fl=[]
    f=0
    seuil=0
    for s in ss:
        y_pred=decision(predict,s)
        f1=metrics.f1_score(label, y_pred, average='macro')
        fl.append(f1)
        if(f<f1):
            f=f1
            seuil=s
    return ss,fl,f,seuil

import numpy as np
import matplotlib.pyplot as plt
def plot_threshold(ss,fl,f,seuil):
    plt.plot(ss, fl, label='seuil') #blue
    plt.plot([seuil], [f],'rx',markersize=6) #red
    plt.xlabel("seuil")
    plt.ylabel("F1-score")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show() # affiche la figure a l'ecran

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

# %% [markdown]
# ############ Generate Data #####################
# ################################################

# %%
print("====================\nChronicle DB generation\n====================\n")
# vocabulary size: 50 // mean length of the sequences: 8 // pattern lenght: 8
generator=chrodb_generator(nbitems=50,l=8, lp=8)

# pat_gen=chrodb_generator.pattern_generator

# sequences: stores all the chronicle sequences generated by the generator (100 sequences and 30 patterns)
# TODO: Why the the mean lenght of the sequence (l) is 25??
sequences = generator.generate(nb=100, l=25, npat=30, th=0.5)

print("======== PATTERNS =======")

# TODO: figure out difference between objects of type Chronicle stored in DB_c and DB_ch
# DB_c: stores the (30) chronicles generated by seq_generation.chronicle_generator.py
DB_c=[ch for ch in generator.all_patterns()]

# DB_ch: stores the chronicles generated by recognition.chronicle_recognition.py
DB_ch=[affectation(ch) for ch in generator.all_patterns()]

# DB_seq: stores all the sequences taken from each chronicle sequence of the variable sequences
DB_seq=[s.seq for s in sequences]
    
# Removes from DB_seq all events equal to -1
DB_seq=filterseq(DB_seq)

# DB_seq_label: stores the labels for each event in DB_seq. All labels in DB_seq_label are equal to 1
DB_seq_label=makelabel(DB_seq)

assert (len(DB_seq)==len(DB_seq_label))

print("======== make_noise in CH DB =======")
per=.3

# DB_c is split into 2 parts according to the percentage per
# c: is non disturbed data
# ch: is disturbed data
c,ch=split_db_ch(DB_c,per=per)

print("======== p sequence generation =======") 
# TODO: This is the generation of disturbed data??? The parameter pert in generator.generate is not really used,
# so what is the difference between p_seq0 and p_seq1???

# p_se0: stores all the chronicle sequences generated by the generator (300 sequences and len(ch) patterns). This variable is of the same type of the variable sequences defined above
p_se0=generator.generate(nb=300, l=15, npat=len(ch), th=.5,patterns=ch,pert=0)
# p_seq0: stores all the sequences taken from each chronicle sequence of the variable p_se0
p_seq0=[s.seq for s in p_se0]

# p_se1: stores all the chronicle sequences generated by the generator (200 sequences and len(ch) patterns). This variable is of the same type of the variables sequences and p_se0 defined above
# TODO: Why the the mean lenght of the sequence (l) is 25 for p_se1 and 15 for p_se0??
p_se1=generator.generate(nb=200, l=25, npat=len(ch), th=.5,patterns=ch,pert=1)
# p_seq1: stores all the sequences taken from each chronicle sequence of the variable p_se1
p_seq1=[s.seq  for s in p_se1]

# Removes from p_seq0 and p_seq1 all events equal to -1
p_seq0=filterseq(p_seq0)
p_seq1=filterseq(p_seq1)

# Serialize all variables
serialization2(p_seq0,"p_seq0.txt")
serialization(p_seq1,"p_seq1.txt")
serialization2(DB_seq,"DB_seq.txt")
serialization(DB_seq_label,"DB_seq_label.txt")
serialization(DB_ch,"DB_ch.txt")

# %%
############### generate train seq for LSTM ######################
trainseq=[]
m=[25]
for i,ch in enumerate(DB_c[:]):
    for k in m: 
        ps=generator.generate(nb=5, l=k, npat=len([ch]), th=.5,patterns=[ch],pert=-2)
        # pseq=filterseq([s.seq for s in ps])
        pseq=[s.seq for s in ps]
        pseq=filterseq(pseq)
        # print(len(ps))
        if len(pseq)!=0:
            trainseq.extend(pseq)


# %%

serialization(trainseq,"test.txt")
serialization2(trainseq,"test2.txt")

# %%
# read_text_file("p_seq1.txt")

# %%
from LSTM.LSTMtrainBin import *
print("####### Start of traning ########")

h=train(p_seq1)
saveobj("history.hist",h)

print("####### End of traning ########")

# %%
##load model and parameter
# from keras.models import load_model

model = load_model('./LSTM/output_files/model.h5')
with open('./LSTM/output_files/config.dictionary', 'rb') as config_file:
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

print("======== SEQUENCES =======")
seq_df=pd.DataFrame({'sequence':[], 'label':[]})
#DB_seq=[]
#DB_seq_label=[]
seq_df.sequence=pd.Series(DB_seq[:]+p_seq0[:]+p_seq1[:])
nbseq=len(seq_df.sequence)

seq_df.label=pd.Series(len(DB_seq[:])*[1]+len(p_seq0[:])*[1]+len(p_seq1[:])*[0])
print(seq_df.shape)
print(seq_df.head(10))

# %%

def LSTMpred(seqs):
    start_time = time.perf_counter()
    SS=[anomalydetect([seq],param,model) for seq in [filterseq(seqs)[0]]]
    end_time=time.perf_counter()
    le=end_time-start_time
    return SS,le 

SS,le= LSTMpred(list(seq_df.sequence))
print(SS)

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
#DB_seq=[]
#DB_seq_label=[]
seq_df.sequence=pd.Series(DB_seq[:]+p_seq0[:]+p_seq1[:])
nbseq=len(seq_df.sequence)

seq_df.label=pd.Series(len(DB_seq[:])*[1]+len(p_seq0[:])*[1]+len(p_seq1[:])*[0])
print(seq_df.shape)
print(seq_df.head(10))

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








