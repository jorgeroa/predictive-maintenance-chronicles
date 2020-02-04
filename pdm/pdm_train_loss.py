
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from collections import Counter
import unicodecsv
import numpy as np
import random
import sys
import os
import copy
import csv
import time
import pickle
import category_encoders as ce
import pandas as pd

from data_utils import *

from datetime import datetime
from math import log

from keras import backend as K
print((K.tensorflow_backend._get_available_gpus()))

import keras
import tensorflow as tf
config = tf.ConfigProto(device_count={'GPU':1})
sess = tf.Session(config = config)
K.set_session(sess)

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# if __name__ == "__main__":
#     #from LSTMpredict import *
#     from pdm_monitoring import *
# else:
#     #from .LSTMpredict import *
#     from .pdm_monitoring import *

import re 

print("TF VERSION: ",tf.__version__)

class modelparameter:
    def __init__(self,divisor,divisor2,maxlen,target_chars,target_char_indices,target_indices_char,char_indices,chars):
        self.divisor,self.divisor2,self.maxlen,self.target_chars,self.target_char_indices,self.target_indices_char,self.char_indices,self.chars=divisor,divisor2,maxlen,target_chars,target_char_indices,target_indices_char,char_indices,chars
    def __str__(self):
        return '{0}{1}{2}{3}{4}{5}{6}{7}'.format(self.divisor,self.divisor2,self.maxlen,self.target_chars,self.target_char_indices,self.target_indices_char,self.char_indices,self.chars)

# from keras.utils import losses_utils
from keras.layers import Lambda
from keras.layers import add
# from keras.losses import LossFunctionWrapper

# class ProbLikelihood(LossFunctionWrapper):
#     """Computes the mean of squares of errors between labels and predictions.
#     Standalone usage:
#     ```python
#     mse = keras.losses.MeanSquaredError()
#     loss = mse([0., 0., 1., 1.], [1., 1., 1., 0.])
#     ```
#     Usage with the `compile` API:
#     ```python
#     model = keras.Model(inputs, outputs)
#     model.compile('sgd', loss=keras.losses.MeanSquaredError())
#     ```
#     # Arguments
#         reduction: (Optional) Type of loss reduction to apply to loss.
#             Default value is `SUM_OVER_BATCH_SIZE`.
#         name: (Optional) name for the loss.
#     """

#     def __init__(self,
#                 #  reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
#                  name='prob_likelihood'):
#         super(ProbLikelihood, self).__init__(
#             prob_likelihood, name=name, reduction=reduction)

def prob_likelihood(t1,t2):
    #lamda=0.0000015
    lamda=0.00045
    # Lambda for subtracting two tensors
    minus_t2 = Lambda(lambda x: -x)(t2)
    subtracted = add([t1,minus_t2])
    p=K.exp(-lamda*(K.abs(subtracted)))
    return p
            
  
def extract(input):  
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
    return [ s for s in l if len(s)!=0]

def train(DB_seq):

    lines = [] #these are all the activity seq
    timeseqs = [] #time sequences (differences between two events)
    timeseqs2 = [] #time sequences (differences between the current and first)
        
    times = []
    times2 = []
    evnts=[]

    numlines= len(DB_seq)
    for seq in DB_seq: #the rows are "ChID,sequence,TC"
        lastevnettime=seq[0][0]
        # lastevnettime=round(seq[0][0]/60)
        firsteventtime=seq[0][0]
        # firsteventtime=round(seq[0][0]/60)

        times = []
        times2 = []
        evnts=[] 
        for t,e in seq:
            # JGT: I added this line to get hours instead of minutes and hence improve time prediction
            # t=round(t/60)
            evnts.append(e)
            times.append(t-lastevnettime)
            times2.append(t-firsteventtime)
            lastevnettime=t
        lines.append(evnts)
        timeseqs.append(times)
        timeseqs2.append(times2)
        
    print("e: ",lines)
    print("t: ", timeseqs)

    ########################################

    divisor = np.mean([item for sublist in timeseqs for item in sublist]) #average time between events
    print('divisor: {}'.format(divisor))
    divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist]) #average time between current and first events
    print('divisor2: {}'.format(divisor2))

    #########################################################################################################

    step = 1
    sentences = []
    softness = 0
    next_chars = []
    #lines = [x+'!' for x in lines]
    maxlen = max([len(x) for x in lines]) #find maximum line size

    # next lines here to get all possible characters for events and annotate them with numbers
    chars = [set(x) for x in lines]
    chars = list(set().union(*chars))
    chars.sort()
    target_chars = copy.copy(chars)
    #chars.remove('!')
    print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
    print(indices_char)

    ######### save parameters ###########
    print(divisor,divisor2,maxlen,target_chars,target_char_indices,target_indices_char,char_indices,chars)
    m=modelparameter(divisor,divisor2,maxlen,target_chars,target_char_indices,target_indices_char,char_indices,chars)
    saveobj(f_config,m)
    # with open('./pdm/output_files/config.dictionary', 'wb') as config_file:
    #     pickle.dump(m, config_file)
    
    #####################################
    
    label=["evt"]
    df = pd.DataFrame(chars,columns=label)
    ce_bin = ce.BinaryEncoder(cols=['evt'],drop_invariant=True)
    r=ce_bin.fit_transform(df.evt.to_frame())
    ##############################
    codelines=[]
    sentences_t = []
    next_chars_t = []
    lines_t=timeseqs
    for line, line_t in  zip(lines, lines_t):
        b=ce_bin.transform(pd.DataFrame(line,columns=['evt']))
        bb=b.values.tolist()
        for i in range(0, len(line), step):
            if i==0:
                continue

            #we add iteratively, first symbol of the line, then two first, three...
            codelines.append(bb[0:i])
            sentences.append(line[0: i])
            sentences_t.append(line_t[0:i])
            next_chars.append(line[i])
            if i==len(line)-1: # special case to deal time of end character
                next_chars_t.append(0)
            else:
                next_chars_t.append(line_t[i])

    print('nb sequences:', len(sentences))

    print('Vectorization...')
    num_features = r.shape[1]+2
    print('num features: {}'.format(num_features))
    
    X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
    y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
    y_t = np.zeros((len(sentences)), dtype=np.float32)


    for i, sentence in enumerate(sentences):
        leftpad = maxlen-len(sentence)
        next_t = next_chars_t[i]

        sentence_t = sentences_t[i]
        bb=codelines[i]
        for t, char in enumerate(sentence):
            #multiset_abstraction = Counter(sentence[:t+1])
            X[i, t+leftpad]=bb[t]+[t+1 , sentence_t[t]/divisor] 
            # X[i, t+leftpad]=bb[t]+[t+1 , sentence_t[t]] 

        y_a[i, target_char_indices[next_chars[i]]] = 1-softness

        y_t[i] = next_t/divisor
        # y_t[i] = next_t

        np.set_printoptions(threshold=sys.maxsize)


    # build the model: 
    print('Build model...')
    print(maxlen)
    #keras.backend.get_session().run(tf.global_variables_initializer())
    main_input = Input(shape=(maxlen, num_features), name='main_input')
    # train a 2-layer LSTM with one shared layer
    l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
    b1 = BatchNormalization()(l1)
    l2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(b1) # the shared layer
    b2 = BatchNormalization()(l2)
    # l3 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(b2) # the shared layer
    # b3 = BatchNormalization()(l2)
    l3 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(b2) # the shared layer
    b3 = BatchNormalization()(l3)
    l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b3) # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)
    l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b3) # the layer specialized in time prediction
    b2_2 = BatchNormalization()(l2_2)

    act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

    model = Model(inputs=[main_input], outputs=[act_output, time_output])

    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)




    from math import exp

    # # Define custom loss
    # def custom_loss(layer):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        prob = prob_likelihood(y_pred, y_true)
        minus = Lambda(lambda x: -x)(prob)
        # subtracted = add([1,minus])
        l = K.mean(K.square(1 - minus))
        return l

    # # Return a function
    # return loss

    # # Define custom metric
    # def custom_metric(layer):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def metric(y_true,y_pred):
        return K.mean(prob_likelihood(y_pred, y_true))

    # # Return a function
    # return metric


    model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':loss}, 
                    optimizer=opt, 
                    metrics={'act_output': 'accuracy', 'time_output':[metric,'mse']})
                    # , 
                    # metrics={'act_output': 'accuracy', 'time_output': ['accuracy', 'mse']})

    early_stopping = EarlyStopping(monitor='val_loss', patience=42)
    model_checkpoint = ModelCheckpoint(f_model, 
                                        monitor='val_loss', 
                                        verbose=0, 
                                        save_best_only=True, 
                                        save_weights_only=False, 
                                        mode='auto')
    # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    # JGT:
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.2, 
                                    patience=10, 
                                    verbose=0, 
                                    mode='auto', 
                                    min_delta=0.0001, 
                                    cooldown=0, 
                                    min_lr=0.001)

    # JGT:
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=fold_model_summary,histogram_freq=1)

    print("model.fit()",time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    h=model.fit(X, {'act_output':y_a, 'time_output':y_t}, 
                    validation_split=0.2, 
                    verbose=2, 
                    callbacks=[early_stopping, model_checkpoint, lr_reducer, tensorboard_callback], 
                    batch_size=maxlen, 
                    epochs=500)


    return h
