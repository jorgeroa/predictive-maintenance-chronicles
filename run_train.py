# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from data_utils import *
from pdm_data_generation.pdmdb_generator import *
from pdm.pdm_train import *

from keras.models import load_model

from matplotlib import pyplot as plt


# %%

# # Load sequences generated with the PdM generator
sequences = loadobj(f_seq_normal_bin)
disturbed_sequences = loadobj(f_seq_disturbed_bin)


# %%

# Start training
print("####### Start traning ########")

h=train(sequences)
saveobj(f_model,h)

print("####### End traning ########")


# %%
##load model and parameter

model = load_model(f_model)
with open(f_config, 'rb') as config_file:
    param = pickle.load(config_file)
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
