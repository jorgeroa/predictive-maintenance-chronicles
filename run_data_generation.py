# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from data_utils import *
from pdm_data_generation.pdmdb_generator import *

import pandas as pd

# %%
# ############ LOAD PARAMETERS FOR DATA GENERATION #####################
# ######################################################################
import json

with open('pdm_data_generation/config_generator.json') as json_file:
    config_data = json.load(json_file)

# #########################################
# Parameters setting for the generator
n_events = config_data["nEvents"]   # Number of events
events_per_pattern = config_data["eventsPerPattern"]  # Number of events per pattern
constraint_density = config_data["constraintDensity"]    # Percentaje of connected events that will have constraints (1 means all connections will have constraints, 0 the opposite)
min_start = config_data["minStart"]   # Time of minimun start of an event
min_duration = config_data["minDuration"]    
max_duration = config_data["maxDuration"]     # Max duration for a time constraint (5 hours=60*5=300).
# #########################################


# #########################################
# Parameters setting for the generation of sequences using the generator
n_sequences = config_data["nSequences"]     # Number of sequences to be generated
sequences_mean_lenght = config_data["sequencesMeanLenght"]   # Maximun length of a sequence
n_patterns = config_data["nPatterns"]   # Number of patterns to generate
pattern_coverage = config_data["patternCoverage"]  # Percentaje of sequences covering each pattern
# #########################################


# %%
# ############ Generate Data #####################
# ################################################
# nbitems: number of events // lp: number of events per pattern
generator=pdmdb_generator(nbitems=n_events, 
                            lp=events_per_pattern, 
                            dc=constraint_density, 
                            minstart=min_start, 
                            minduration=min_duration, 
                            maxduration=max_duration)

# sequences: stores all the chronicle sequences generated by the generator 
chro_sequences = generator.generate(nb=n_sequences, 
                                l=sequences_mean_lenght, 
                                npat=n_patterns, 
                                th=pattern_coverage)

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

# sequences: stores all the chronicle sequences generated by the generator 
disturbed_chro_sequences = generator.generate(nb=n_sequences, 
                                l=sequences_mean_lenght, 
                                npat=n_patterns, 
                                th=pattern_coverage,
                                patterns=disturbed_chronicles)

# Stores all the sequences taken from each chronicle sequence of the variable disturbed_chro_sequences
disturbed_sequences=[s.seq for s in disturbed_chro_sequences]

# %%

# Removes from sequences and disturbed_sequences all events equal to -1 and empty sequences
sequences=filterseq(sequences)
disturbed_sequences=filterseq(disturbed_sequences)

# %% [markdown]
# ############ Saves Data #####################
# #############################################

#  %%

# ###################### SERIALIZATION #########################
# saveobj("generator{0}/bin_seqs/seq{0}.dict".format(iexp,iseq),variable_name)

# ############# SAVE GENERATORS #################

saveobj(f_generator_bin,generator)
saveobj(f_chronicles_bin,chro_sequences)
saveobj(f_disturbed_chronicles_bin,disturbed_chro_sequences)

# ############# SAVE SEQUENCES #################
saveobj(f_seq_normal_bin,sequences)
saveobj(f_seq_disturbed_bin,disturbed_sequences)

# Serialize sequences of events
serialization2(sequences,f_seq_normal_txt)
serialization2(disturbed_sequences,f_seq_disturbed_txt)

# Serialize sequences of events with time
serialization2(sequences,f_seq_normal_time_txt)
serialization2(disturbed_sequences, f_seq_disturbed_time_txt)

# %% [markdown]
# ############ LOAD DATA #####################
# #############################################
# %%

sequences = loadobj(f_seq_normal_bin)
disturbed_sequences = loadobj(f_seq_disturbed_bin)


# %%

to_csv(sequences, f_seq_normal_csv)

# %% [markdown]

# ######### ANALIZE GENERATED DATA ##########


# %%

# ############### Data Frame #################
# ###########################################

# nbitems=50 #nb vocab
# seqlen=10 #len ch
# # p_seq0=[]
# # p_seq1=[]
# DB_seq=sequences[:]

# # DB_seq[:]+p_seq0[:]+p_seq1[:]

# times = []
# events=[]

# for i,seq in enumerate(DB_seq): #the rows are "ChID,sequence,TC"
#     if len(seq)==0:
#         continue
#     for t,e in seq:
#         events.append(e)
#         times.append(t)

#     # lines.append(evnts)
#     # timeseqs.append(times)

# print(len(events))
# print(len(times))
# # seq_df=pd.DataFrame({'event':events, 'time':times})
# seq_df=pd.DataFrame({'event':events, 'time':times})

# seq_df.head()

# # s=seq_df.event=pd.Series(events)
# # seq_df.time=pd.Series(times)

# # # seq_df.sequence=pd.Series(DB_seq[:]+p_seq0[:]+p_seq1[:])

# # s.value_counts()
# # # seq_df.plot(kind='bar')


# %%

# ############### DATA ANALYSIS #################
# ###############################################

from matplotlib import pyplot

from pandas import read_csv
from pandas import Grouper
from pandas import DataFrame
from pandas import to_datetime

import numpy as np

#   Load data from csv
series = read_csv(f_seq_normal_csv, header=0, index_col=0)

#  %%

series.hist()
pyplot.show()

# %%

groups = series.groupby('Event')
# groups.first()}
events = DataFrame()

for name, group in groups:
    events[name] = pd.Series(np.asarray(group['Event'].index))

events.plot(subplots=True, legend=True)
pyplot.show()


# %%

# Reload data without index=0
series = read_csv(f_seq_normal_csv, header=0)

series.plot(kind='kde')
pyplot.show()
print("Density of events")
series['Event'].plot(kind='kde')
pyplot.show()
print("Density of time")
series['Time'].plot(kind='kde')
pyplot.show()
#  %%

groups = series.groupby('Event')
# groups.first()}
events = DataFrame()

for name, group in groups:
    events[name] = pd.Series(np.asarray(group['Event'].index))

events.boxplot()
pyplot.show()

# %%
groups = series.groupby('Event')
# groups.first()}
events = DataFrame()

for name, group in groups:
    events[name] = pd.Series(np.asarray(group['Event'].index))

events = events.T
pyplot.matshow(events, interpolation=None, aspect='auto')
pyplot.show()


# %%

from pandas.plotting import lag_plot

print("Lag plot Events")
lag_plot(series["Event"])
pyplot.show()

print("Lag plot Time")
lag_plot(series["Time"])
pyplot.show()

print("Lag plot Relative Time")
lag_plot(series["Time"]/60)
pyplot.show()

# %%

from pandas.plotting import lag_plot

groups = series.groupby('Event')
# groups.first()}
events = DataFrame()

for name, group in groups:
    events[name] = pd.Series(np.asarray(group['Event'].index))
    print("Lag plot Event for event {0}".format(name))
    lag_plot(events[name])
    pyplot.show()

# %%

from pandas.plotting import lag_plot

rel_time = [int(i/60) for i in series['Time']]

groups = series.groupby('Event')
# groups.first()}
events = DataFrame()

for name, group in groups:
    rel_time = [int(i/60) for i in np.asarray(group['Event'].index)]
    events[name] = pd.Series(rel_time)
    print("Lag plot Event for event {0}".format(name))
    lag_plot(events[name])
    pyplot.show()

# %%

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(series)
pyplot.show()

print("Autocorrelation of Events")
autocorrelation_plot(series['Event'])
pyplot.show()

print("Autocorrelation of Time")
autocorrelation_plot(series['Time'])
pyplot.show()

# %%

from pandas.plotting import lag_plot

rel_time = [int(i/60) for i in series['Time']]

groups = series.groupby('Event')
# groups.first()}
events = DataFrame()

for name, group in groups:
    rel_time = [int(i/60) for i in np.asarray(group['Event'].index)]
    events[name] = pd.Series(rel_time)
    print("Lag plot Event for event {0}".format(name))
    autocorrelation_plot(events[name])
    pyplot.show()
# %%

print("======== SEQUENCES =======")
seq_df=pd.DataFrame({'sequence':[], 'label':[]})
seq_df.sequence=pd.Series(sequences[:])
# nbseq=len(seq_df.sequence)

seq_df.label=pd.Series(len(sequences[:])*[0])
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







