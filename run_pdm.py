# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from data_utils import *
from pdm_data_generation.pdmdb_generator import *

from pdm.pdm_train  import *
from pdm.pdm_monitoring import *


# %%
##load model and parameter
# from keras.models import load_model

model = load_model('./pdm/output_files/model.h5')
with open('./pdm/output_files/config.dictionary', 'rb') as config_file:
    param = pickle.load(config_file)

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

def LSTMpred(seqs):
    start_time = time.perf_counter()
    SS=[anomalydetect([seq],param,model) for seq in seqs]
    end_time=time.perf_counter()
    le=end_time-start_time
    return SS 

result=LSTMpred(list(seq_df.sequence))
# print(result)
save_text_file(result,"result.txt")





# %%
