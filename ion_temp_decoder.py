# -*- coding: utf-8 -*-
# %% 
"""
Mapping ion temperature profiles to X-ray images
Joel Aftreth, Max Planck Institute for Plasma Physics, Greifswald DE
"""

import tensorflow as tf
import random
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import tensorflow_addons as tfa 
import gc
from sewar.full_ref import rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import torch
from matplotlib.gridspec import GridSpec
# %% 
physical_devices = tf.config.list_physical_devices()
print("DEVICES : \n", physical_devices)
print(tf.config.list_physical_devices('GPU'))

print('Using:')
print('\t\u2022 Python version:',sys.version)
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if len(tf.config.list_physical_devices('GPU'))>0 else '\t\u2022 GPU device not found. Running on CPU')

random.seed(123)
CNN = False
BATCH_SIZE = 256
processed_ds = False
test_model = True
plot_only = True
load_pretrained = False
if plot_only:
    load_pretrained = True
if test_model:
    test_str = '_test'
else:
    test_str = ''
if processed_ds:
    process_str = 'processed'
else:
    process_str = 'unprocessed'

base_dir = r"C:\Users\joaf\Documents\models"
MODEL_FNAME = base_dir+f"\\decoder_model_{process_str}{test_str}.h5"
save_plots_dir = r"C:\Users\joaf\Documents\results\decoder"
saved_model = base_dir+f'\\decoder_model_{process_str}{test_str}.h5'

beta_1 = 0.1
learning_rate = 1e-2
wd = 0 

num_epochs = 100
num_patience = int(num_epochs * 0.3)
 
#%%
# Define the input and output shapes
if processed_ds:
    input_shape = (1, 15)
    output_shape = (20, 46, 1)
else:    
    input_shape = (1, 15)
    output_shape_raw = (20, 195, 1)
    output_shape = (20, 46, 1)


# Define the training and testing datasets
if processed_ds:
    hdf5_path = r'\\x-drive\Diagnostic-logbooks\Minerva\XICS\training_data_set\process_ds.h5'
else:
    hdf5_path = r'\\x-drive\Diagnostic-logbooks\Minerva\XICS\training_data_set\merged_ds.h5'

def extract_mean_var(x, axes=0): # originally axes=[0,1]
    mean, variance = tf.nn.moments(x, axes=axes)
    return mean, variance

with h5py.File(hdf5_path, 'r') as hf:
    for group in hf:
        group_key = group
        if processed_ds:
            test_ds_key = list(hf[group_key].keys())[0]
            train_ds_key = list(hf[group_key].keys())[1]
            target_key = list(hf[group_key][train_ds_key].keys())[0]
            input_key = list(hf[group_key][train_ds_key].keys())[2]
            test_target_key = list(hf[group_key][test_ds_key].keys())[0]
            test_input_key = list(hf[group_key][test_ds_key].keys())[1]
            train_num_dset_inputs = hf[group_key][train_ds_key][input_key].shape[0]
            train_num_dset_targets = hf[group_key][train_ds_key][target_key].shape[0]
            test_num_dset_inputs = hf[group_key][test_ds_key][test_input_key].shape[0]
            test_num_dset_targets = hf[group_key][test_ds_key][test_target_key].shape[0]
            
            
            train_input_example = hf[group_key][train_ds_key][input_key][0]
            train_target_example = hf[group_key][train_ds_key][target_key][0]
            test_input_example = hf[group_key][test_ds_key][test_input_key][0]
            test_target_example = hf[group_key][test_ds_key][test_target_key][0]
        else:    
            # #merged_ds
            target_key = list(hf[group_key].keys())[0] #target meaning ion_temp profiles
            input_key = list(hf[group_key].keys())[1] #input meaning x-ray images
            train_input_example = hf[group_key][input_key][0]
            train_target_example = hf[group_key][target_key][0]
            merged_num_dset_inputs = hf[group_key][input_key].shape[0]
            merged_num_dset_targets = hf[group_key][target_key].shape[0]
            
            # max and min values over whole dataset, takes a while to calculate, so hard coded        
            max_train = 9915825.697197208   
            min_train = -4.279900956759586
def prime_factors(n):
    """
    Returns a list of the prime factors of the input integer n.
    """
    factors = []
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
def make_equal_length(list1, list2):
    if len(list1) < len(list2):
        list1 += [1] * (len(list2) - len(list1))
    elif len(list2) < len(list1):
        list2 += [1] * (len(list1) - len(list2))
    return list1, list2

output_factorized_0, output_factorized_1 = prime_factors(output_shape[0]),prime_factors(output_shape[1])
output_factorized_0, output_factorized_1 = make_equal_length(output_factorized_0, output_factorized_1)
#%%
num_conv_layers = 1
start_shape = (int(np.ceil((output_shape[0]/2**num_conv_layers))), int(np.ceil(output_shape[1]/2**num_conv_layers)))
if CNN:
    def make_generator_model():
        if not processed_ds:
            reshape_layer = tf.keras.layers.Lambda(lambda x: x)
        else:
            reshape_layer = tf.keras.layers.Lambda(lambda x: x)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(np.prod(start_shape), activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Reshape((start_shape[0], start_shape[1],1)),
            tf.keras.layers.LocallyConnected2D(16,kernel_size=(3,3),strides=(1,1),padding='same',implementation=2,\
                kernel_regularizer=tf.keras.regularizers.l2(0.001), activation = tf.keras.layers.LeakyReLU()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2,2), padding='same', \
                kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2,2), padding='same', \
                kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2,2), padding='same', \
                kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(32,kernel_size=(3,3),strides=(2,2),padding='same',\
                kernel_regularizer=tf.keras.regularizers.l2(0.001), implementation=2,activation = tf.keras.layers.LeakyReLU()),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Conv2D(64,kernel_size=(3,3),strides=(2,2),padding='same',\
                kernel_regularizer=tf.keras.regularizers.l2(0.001), implementation=3,activation = tf.keras.layers.LeakyReLU()),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Conv2D(1,kernel_size=1, strides= 1, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            reshape_layer,
            tf.keras.layers.Reshape(output_shape),
            tf.keras.layers.Activation('tanh'),
        ])
        return model
if not CNN:
    def make_generator_model():
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_shape=input_shape, activation='gelu'))
        model.add(tf.keras.layers.Dense(256, input_shape=input_shape, activation='gelu'))
        model.add(tf.keras.layers.Dense(512, activation='gelu'))
        model.add(tf.keras.layers.Dense(np.prod(output_shape), activation='linear'))
        model.add(tf.keras.layers.Reshape(output_shape))
        return model
    
model = make_generator_model()


if plot_only and load_pretrained:
    model = tf.keras.models.load_model(saved_model)
elif load_pretrained:
    model = tf.keras.models.load_model(saved_model)
model.summary()
#%%
# Define the loss function and optimizer
loss_fn = tf.keras.losses.MeanAbsoluteError()

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
@tf.function
def custom_loss(predicted_images, target_images):
    losses = tf.image.ssim(tf.convert_to_tensor(predicted_images),target_images,2)
    return tf.reduce_mean(losses)

optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=wd, beta_1 = beta_1)

# Compile the model
model.compile(loss=loss_fn, optimizer=optimizer,run_eagerly=True)

#%%

# Define the input and output shapes
input_shape = (input_shape[0], input_shape[1])
output_shape = (output_shape[0], output_shape[1])

class generator:
    def __init__(self, file, group_key, input_key, target_key, train_on):
        self.file = file
        self.group_key = group_key
        self.input_key = input_key
        self.target_key = target_key
        self.train_on = train_on
        if self.train_on:
            if processed_ds:
                self.i = random.randrange(train_num_dset_inputs)
            else: 
                self.i = random.randrange(merged_num_dset_inputs)
        else:
            if processed_ds:
                self.i = random.randrange(test_num_dset_inputs)
            else: 
                self.i = random.randrange(merged_num_dset_inputs)
    
    def __del__(self):
      # Close the HDF5 file handle when the generator is deleted.
      hf.close()
        
    if processed_ds:    
        def __call__(self):    
            with h5py.File(self.file, 'r') as hf:
                while True:
                    if self.train_on:
                        try:
                            in_im = hf[self.group_key][train_ds_key][self.input_key][self.i]
                            in_im = np.reshape(in_im, input_shape, order='C') #shape of (1,15)
                            out_im = hf[self.group_key][train_ds_key][self.target_key][self.i]
                            
                            #crop to  first 47 wavelengths, look at only middle lines of sight (for plots)
                            # use original ion_temp profiles
                            out_im = np.reshape(out_im, output_shape, order='C')
                            
                            yield in_im, out_im
                        except GeneratorExit as e:
                            print(f"Caught GeneratorExit exception {e}. Closing HDF5 file.")
                            break
                        finally:
                            if self.i < train_num_dset_inputs -1:
                                self.i += 1
                            else: 
                                self.i = 0 
                    else:
                        try:
                            in_im = hf[self.group_key][test_ds_key][self.input_key][self.i]
                            in_im = np.reshape(in_im, input_shape, order='C') #shape of (1,15)
                            out_im = hf[self.group_key][test_ds_key][self.target_key][self.i]
                            out_im = np.reshape(out_im, output_shape, order='C')#shape of (20,195)
                            yield in_im, out_im
                        except GeneratorExit as e:
                            print(f"Caught GeneratorExit exception {e}. Closing HDF5 file.")
                            break
                        finally:
                            if self.i < test_num_dset_inputs -1:
                                self.i += 1
                            else: 
                                self.i = 0 
                    
    else:
        def __call__(self):    
            with h5py.File(self.file, 'r') as hf:
                while True:
                    if self.train_on:
                        try:
                            in_im = hf[self.group_key][self.input_key][self.i]
                            in_im = np.reshape(in_im, input_shape, order='C') #shape of (1,15)
                            out_im = hf[self.group_key][self.target_key][self.i]
                            #normalize to [0,1]
                            out_im = (out_im - np.min(out_im)) / (np.max(out_im) - np.min(out_im)) 
                            out_im = np.reshape(out_im, output_shape_raw, order='C')
                            out_im = out_im[:,:output_shape[1],0]
                            yield in_im, out_im
                        except GeneratorExit as e:
                            print(f"Caught GeneratorExit exception {e}. Closing HDF5 file.")
                            break
                        finally:
                            if self.i < merged_num_dset_inputs -1:
                                self.i += 1
                            else: 
                                self.i = 0   
                    else:
                        try:
                            in_im = hf[self.group_key][self.input_key][self.i]
                            in_im = np.reshape(in_im, input_shape, order='C') #shape of (1,15)
                            out_im = hf[self.group_key][self.target_key][self.i]
                            #normalize to [0,1]
                            out_im = (out_im - np.min(out_im)) / (np.max(out_im) - np.min(out_im)) 
                            out_im = np.reshape(out_im, output_shape_raw, order='C')
                            out_im = out_im[:,:output_shape[1],0]
                            yield in_im, out_im
                        except GeneratorExit as e:
                            print(f"Caught GeneratorExit exception {e}. Closing HDF5 file.")
                            break
                        finally:
                            if self.i < merged_num_dset_inputs -1:
                                self.i += 1
                            else: 
                                self.i = 0                  

# Create the dataset using the generator function
train_dataset = tf.data.Dataset.from_generator(
    generator(hdf5_path,group_key, input_key, target_key,True),
    output_types=(tf.float32, tf.float32),
    output_shapes=(input_shape, output_shape)
)
test_dataset = tf.data.Dataset.from_generator(
    generator(hdf5_path,group_key, input_key, target_key,False),
    output_types=(tf.float32, tf.float32),
    output_shapes=(input_shape, output_shape)
)

train_dataset = train_dataset.repeat().batch(BATCH_SIZE)
test_dataset = test_dataset.repeat().batch(BATCH_SIZE)

""" 
Train the model
"""
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_FNAME, verbose=2, monitor='loss', mode='min', save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=num_patience,restore_best_weights=True)
class BestEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BestEpochCallback, self).__init__()
        self.best_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch
        print(f"Best epoch so far: {self.best_epoch + 1}")
class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            gc.collect()
            tf.keras.backend.clear_session()


#%% TRAIN AND PLOT LOSS       
if not plot_only:
    history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch = 10,
                        callbacks = [checkpointer, early_stopping,BestEpochCallback(), ClearMemory()])

    plt.plot(history.history['loss'][:])
    plt.title('model loss: MAE')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show(block=False)


# %% PLOTS
##################################################################

num_plots = 4 
rhos = np.linspace(0,1,input_shape[1])
fig, axs = plt.subplots(figsize=(10,16),nrows=num_plots, ncols = 3, gridspec_kw={'width_ratios': [1, 1, 1]}, tight_layout =True)
plt.suptitle(f"{process_str.capitalize()} X-Ray Image Predictions from Ion Temp Profile")
i = 0
for inputs, outputs in test_dataset.take(num_plots): 
    images = model.predict(inputs)
    
    axs[i][0].plot(rhos, inputs[0,0,:], label='input',color = 'orange')
    axs[i][0].set_title(f'input profile')
    axs[i][0].set_ylabel('temp (keV)')
    axs[i][0].set_xlabel('rho')
    im1 = axs[i][1].imshow(outputs[0,:,:],vmin=0, vmax=1)
    #add color bars
    axs[i][1].set_title(f'true image')
    axs[i][1].set_ylabel('line of sight')
    axs[i][1].set_xlabel('wavelength')
    fig.colorbar(im1, ax=axs[i][1])
    im2 = axs[i][2].imshow(images[0,:,:],vmin=0, vmax=1)
    error = model.evaluate(inputs,outputs, steps=1, batch_size=1)
    axs[i][2].set_title(f'prediction with MAE: {error:.1f}')
    axs[i][2].set_ylabel('line of sight')
    axs[i][2].set_xlabel('wavelength')
    fig.colorbar(im2, ax=axs[i][2])
    i += 1
plt.savefig(save_plots_dir+f"\\decoder_predictions_{process_str}{test_str}.png")
    
    
from matplotlib import gridspec 

fig = plt.figure(figsize = (10,20),constrained_layout=True)
plt.suptitle(f"{process_str.capitalize()} X-Ray Image Predictions from Ion Temp Profile")
gs = fig.add_gridspec(num_plots*4,6)
i = 0
for inputs, outputs in test_dataset.take(num_plots): 
    images = model.predict(inputs)
    axbig = plt.subplot(gs[i*4:i*4+4, 0])
    axbig.plot(rhos, inputs[0,0,:], label='input',color = 'orange')
    axbig.set_title(f'input profile')
    axbig.set_ylabel('temp (keV)')
    axbig.set_xlabel('rho')
    for row in np.arange(0,4):
        for col in np.arange(0,5):
            ax = plt.subplot(gs[i*4+row, col +1])
            ax.plot(outputs[0,int(row*5 +col),:], label ='true',color = 'orange')
            ax.plot(images[0,int(row*5 +col),:,0], label ='prediction',color = 'blue')
            ax.set_title(f'LOS {int(row*5 +col +1)}')
            if row == 0 and col == 4:
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    i+=1
plt.savefig(save_plots_dir+f"\\decoder_predictions_{process_str}{test_str}_LOS.png")

