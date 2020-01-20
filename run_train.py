# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from data_utils import *
from pdm_data_generation.pdmdb_generator import *
from pdm.pdm_train import *

from keras.models import load_model

from matplotlib import pyplot as plt

# %%

# Set path variables
ofile='./experiments/output_files/data/%s'
ifile='./experiments/input_data/%s'
model_path='./experiments/model/%s'
model_name='model.h5'
config_name='config.dictionary'

# %%

# Load sequences generated with the PdM generator
sequences = loadobj(ofile% "sequences.pick")
disturbed_sequences = loadobj(ofile% "disturbed_sequences.pick")


# %%

# Start training
print("####### Start traning ########")

h=train(sequences)
saveobj(model_path% "history.hist",h)

print("####### End traning ########")


# %%
##load model and parameter

model = load_model(model_path% model_name)
with open(model_path% config_name, 'rb') as config_file:
    param = pickle.load(config_file)
# %%

# If the model was just trained jump to the next cell. There is no need to reload the history from a file.
# Load history if the model is saved in a file.
history=loadobj(model_path% "history.hist")

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
