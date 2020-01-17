from  recognition.chronicle_recognition import Chronicle
from seq_generation.chronicle_generator import *
from math import pi,sqrt,exp
from random import uniform,sample,shuffle,gauss
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import concurrent.futures
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics


# ############################################################################
# ############################################################################
# ############################################################################

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