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
seq_df.sequence=pd.Series(sequences[:])
# nbseq=len(seq_df.sequence)

seq_df.label=pd.Series(len(sequences[:])*[0])
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

