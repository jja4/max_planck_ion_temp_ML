# -*- coding: utf-8 -*-
# %% 
"""
Mapping ion temperature profiles to X-ray images
Joel Aftreth, Max Planck Institute for Plasma Physics, Greifswald DE
"""

import tensorflow as tf
# from tensorflow.keras import backend as k
# tf.compat.v1.enable_eager_execution()
# import joblib
import random
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import h5py
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_addons as tfa 
import gc
from sewar.full_ref import rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import torch
# from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis
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

# learning_rate = 0.01
# step = tf.Variable(0, trainable=False)
# schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
#     [10000, 15000], [1e-0, 1e-1, 1e-2])
# # lr and wd can be a function or a tensor
# learning_rate = 1e-1 * schedule(step)
# wd = lambda: 1e-4 * schedule(step)
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
            # mean_train, var_train = extract_mean_var(hf[group_key][target_key][:int(1e5)])
            # min_train_idx = tf.argmin(hf[group_key][target_key][:int(1e5)],axis=0)
            # min_train = hf[group_key][target_key][min_train_idx[0]]
            # max_train_idx = tf.argmax(hf[group_key][target_key][:int(1e5)],axis=0)
            # max_train = hf[group_key][target_key][max_train_idx[0]]
            # window = 1e5
            # min_train = np.inf
            # max_train = 0
            # for i in range(int(merged_num_dset_targets/window)):
            #     min_wind = tf.math.reduce_min(hf[group_key][target_key][int(i*window):int((i+1)*window)])
            #     max_wind = tf.math.reduce_max(hf[group_key][target_key][int(i*window):int((i+1)*window)])
            #     if min_wind < min_train:
            #         min_train = min_wind
            #     if max_wind > max_train:
            #         max_train = max_wind    
            #     print(f'finished {i}') 
            
            # max and min values over whole dataset, takes a while to calculate, so hard coded        
            max_train = 9915825.697197208   
            min_train = -4.279900956759586
# Define the decoder convolutional neural network model
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
# def make_generator_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=input_shape),
#         tf.keras.layers.Reshape((1, 1, input_shape[1])),
#         tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(output_factorized_0[0], output_factorized_1[0]), padding='same', activation=None),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.LeakyReLU(),
#         tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(output_factorized_0[1], output_factorized_1[1]), padding='same', activation=None),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.LeakyReLU(),
#         tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(output_factorized_0[2], output_factorized_1[2]), padding='same', activation=None),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.LeakyReLU(),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(np.prod(output_shape), activation='linear'),
#         tf.keras.layers.Reshape(output_shape)
#     ])
#     return model
#%%
num_conv_layers = 1
start_shape = (int(np.ceil((output_shape[0]/2**num_conv_layers))), int(np.ceil(output_shape[1]/2**num_conv_layers)))
if CNN:
    def make_generator_model():
        if not processed_ds:
            # reshape_layer = tf.keras.layers.Lambda(lambda x: x[:, :, :-1, :])
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
            # tf.keras.layers.LocallyConnected2D(32,kernel_size=(3,3),strides=(2,2),padding='same',implementation=2,activation = tf.keras.layers.LeakyReLU()),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Conv2D(64,kernel_size=(3,3),strides=(2,2),padding='same',\
                kernel_regularizer=tf.keras.regularizers.l2(0.001), implementation=3,activation = tf.keras.layers.LeakyReLU()),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(np.prod(output_shape), activation='tanh'),
            # tf.keras.layers.LocallyConnected2D(1,kernel_size=(5,5),strides=(1,1),padding='same',implementation=2,activation = tf.keras.layers.LeakyReLU()),
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
# loss_fn = tf.keras.losses.Huber(delta=0.5)
# loss_fn = tf.keras.losses.MeanSquaredError()

# with sigmoid, from_logits would be True
# loss_fn= tf.keras.losses.BinaryCrossentropy(from_logits=False)

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# @tf.function
# def custom_loss(predicted_images, target_images):
#     # Calculate ERGAS loss for each image pair
#     # target_images = np.array(target_images)
#     predicted_images = tf.convert_to_tensor(predicted_images)
#     losses = []
#     i=0
#     for tgt_img, pred_img in zip(tf.unstack(target_images), tf.unstack(predicted_images)):
#         loss = ergas(tgt_img, pred_img)
#         losses.append(loss)
#         i += 1
#     # Return mean loss across all image pairs
#     return tf.reduce_mean(tf.stack(losses))
@tf.function
def custom_loss(predicted_images, target_images):
    losses = tf.image.ssim(tf.convert_to_tensor(predicted_images),target_images,2)
    return tf.reduce_mean(losses)
# loss_fn=tf.keras.losses.CosineSimilarity()

# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=wd, beta_1 = beta_1)

# Compile the model
# model.compile(loss=custom_loss, optimizer=optimizer,run_eagerly=True)
model.compile(loss=loss_fn, optimizer=optimizer,run_eagerly=True)

#%%

# Define the input and output shapes
# input_shape = tf.TensorShape([input_shape[0], input_shape[1]])
# output_shape = tf.TensorShape([output_shape[0], output_shape[1]])
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
                            
                            # out_im = np.reshape(out_im, (output_shape[0],output_shape[1],1), order ='C')
                            # out_im = tf.image.per_image_standardization(out_im)
                            out_im = np.reshape(out_im, output_shape, order='C')
                            
                            yield in_im, out_im
                        except GeneratorExit as e:
                            # print(f"Caught GeneratorExit exception {e}. Closing HDF5 file.")
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
                            # out_im = np.reshape(out_im, (output_shape[0],output_shape[1],1), order ='C')
                            # out_im = tf.image.per_image_standardization(out_im)
                            out_im = np.reshape(out_im, output_shape, order='C')#shape of (20,195)
                            yield in_im, out_im
                        except GeneratorExit as e:
                            # print(f"Caught GeneratorExit exception {e}. Closing HDF5 file.")
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
                            # out_im = (out_im - np.min(out_im)) / (max_train - min_train) 
                            #standardize to [ mean 0, std 1]
                            # out_im = np.reshape(out_im, (output_shape_raw[0],output_shape_raw[1],1), order ='C')
                            # out_im = tf.image.per_image_standardization(out_im)
                            out_im = np.reshape(out_im, output_shape_raw, order='C')
                            out_im = out_im[:,:output_shape[1],0]
                            yield in_im, out_im
                        except GeneratorExit as e:
                            # print(f"Caught GeneratorExit exception {e}. Closing HDF5 file.")
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
                            # out_im = (out_im - np.min(out_im)) / (max_train - min_train) 
                            #standardize to [ mean 0, std 1]
                            # out_im = np.reshape(out_im, (output_shape_raw[0],output_shape_raw[1],1), order ='C')
                            # out_im = tf.image.per_image_standardization(out_im)
                            out_im = np.reshape(out_im, output_shape_raw, order='C')
                            out_im = out_im[:,:output_shape[1],0]
                            yield in_im, out_im
                        except GeneratorExit as e:
                            # print(f"Caught GeneratorExit exception {e}. Closing HDF5 file.")
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
# Train the model
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
    # plt.plot(history.history['val_loss'][1:])
    plt.title('model loss: MAE')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    # plt.savefig(save_plots_dir+r"\loss.png")
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
    
    # for _ in range(2):#3 to turn the image 90 degrees three times = 270 degrees
    #     images = np.rot90(images)
    #     outputs = np.rot90(outputs)
    axs[i][0].plot(rhos, inputs[0,0,:], label='input',color = 'orange')
    axs[i][0].set_title(f'input profile')
    axs[i][0].set_ylabel('temp (keV)')
    axs[i][0].set_xlabel('rho')
    # axs[i][1].imshow(tf.transpose(outputs[0,:,:]))
    im1 = axs[i][1].imshow(outputs[0,:,:],vmin=0, vmax=1)
    #add color bars
    axs[i][1].set_title(f'true image')
    axs[i][1].set_ylabel('line of sight')
    axs[i][1].set_xlabel('wavelength')
    fig.colorbar(im1, ax=axs[i][1])
    # axs[i][2].imshow(tf.transpose(images[0,:,:,0]))
    im2 = axs[i][2].imshow(images[0,:,:],vmin=0, vmax=1)
    error = model.evaluate(inputs,outputs, steps=1, batch_size=1)
    axs[i][2].set_title(f'prediction with MAE: {error:.1f}')
    axs[i][2].set_ylabel('line of sight')
    axs[i][2].set_xlabel('wavelength')
    fig.colorbar(im2, ax=axs[i][2])
    # axs[i][2].legend(loc='upper left',bbox_to_anchor=(0.8,1.24))
    i += 1
plt.savefig(save_plots_dir+f"\\decoder_predictions_{process_str}{test_str}.png")
    
    
from matplotlib import gridspec 

# fig , fig_axs= plt.subplots(nrows=num_plots*4, ncols = 6,constrained_layout=True)
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
# gs.tight_layout(fig)
plt.savefig(save_plots_dir+f"\\decoder_predictions_{process_str}{test_str}_LOS.png")
#%%
# from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
# blur = images
# org = outputs
# print("MSE: ", mse(blur,org))
# print("RMSE: ", rmse(blur, org))
# # print("PSNR: ", psnr(int(blur), int(org)))
# # print("SSIM: ", ssim(int(blur), int(org)))
# print("UQI: ", uqi(blur, org))
# # print("MSSSIM: ", msssim(blur, org))
# print("ERGAS: ", ergas(blur, org))
# print("SCC: ", scc(blur, org))
# print("RASE: ", rase(blur, org))
# print("SAM: ", sam(blur, org))
# # print("VIF: ", vifp(blur, org))
# %%

# Load pre-trained BigGAN model using TensorFlow Hub
# import tensorflow as tf
# import numpy as np
# import tarfile
# import tensorflow_hub as hub
# tf.compat.v1.reset_default_graph()
# tf.compat.v1.disable_eager_execution()
# import urllib.request

# # Define input and output shapes
# input_shape = (1, 15)
# output_shape = (20, 46)

# module_path = 'https://tfhub.dev/deepmind/biggan-512/2'
# # module = hub.load(module_path)
# module = hub.Module(module_path)

# # # Inspect the input and output signatures of the model
# # input_signature = module.signatures['default'].inputs[0]
# # output_signature = module.signatures['default'].outputs[0]

# # # Extract the input and output shapes from the signatures
# # # input_shape = input_signature.shape.as_list()[1:]
# # # output_shape = output_signature.shape.as_list()[1:]

# # # print("Input shape:", input_shape)
# # # print("Output shape:", output_shape)

# # # Create fine-tuned model

# # # module_path = r'C:\Users\joaf\Documents\biggan'
# # # GAN = hub.load(module_path)
# # fine_tuned_model = tf.keras.Sequential([
# #     tf.keras.layers.InputLayer(input_shape=input_shape),
# #     tf.keras.layers.Dense(128, activation='linear'),
# #     # tf.keras.layers.InputLayer(input_shape=[]),
# #     hub.KerasLayer(module_path,input_shape=[], trainable=False),
# #     # tf.keras.layers.Lambda(lambda x: module.signatures['default'](tf.expand_dims(x, axis=1))),
# #     tf.keras.layers.Dense(np.prod(output_shape), activation='linear'),
# #     tf.keras.layers.Reshape(target_shape=output_shape)
# # ])
# # fine_tuned_model.summary()

# # # Compile fine-tuned model
# # fine_tuned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mae')

# # # Load training dataset
# # # train_inputs = np.load('train_inputs.npy')
# # # train_targets = np.load('train_targets.npy')

# # # Train fine-tuned model
# # history = fine_tuned_model.fit(train_dataset, epochs=10, steps_per_epoch = 10,
# #                     callbacks = [checkpointer, early_stopping,BestEpochCallback()])


# # # Pass input data through the fine-tuned model to get predicted output
# # num_plots = 4 
# # rhos = np.linspace(0,1,input_shape[1])
# # fig, axs = plt.subplots(figsize= (10,16),nrows=num_plots, ncols = 3, gridspec_kw={'width_ratios': [2, 1, 1]}, tight_layout =True)
# # plt.suptitle(f"{process_str.capitalize()} X-Ray Image Predictions from Ion Temp Profile")
# # i = 0
# # for inputs, outputs in test_dataset.take(num_plots): 
# #     images = fine_tuned_model.predict(inputs)
# #     axs[i][0].plot(rhos, inputs[0,0,:], label='input',color = 'orange')
# #     axs[i][0].set_title(f'input profile')
# #     axs[i][0].set_ylabel('temp (keV)')
# #     axs[i][0].set_xlabel('rho')
# #     axs[i][1].imshow(tf.transpose(outputs[0,:,:]))
# #     axs[i][1].set_title(f'true image')
# #     axs[i][1].set_ylabel('line of sight')
# #     axs[i][1].set_xlabel('wavelength')
# #     axs[i][2].imshow(tf.transpose(images[0,:,:,0]))
# #     error = model.evaluate(inputs,outputs, steps=1, batch_size=1)
# #     axs[i][2].set_title(f'prediction with MAE: {error:.1f}')
# #     axs[i][2].set_ylabel('line of sight')
# #     axs[i][2].set_xlabel('wavelength')
# #     # axs[i][2].legend(loc='upper left',bbox_to_anchor=(0.8,1.24))
# #     i += 1

# # %%
# truncation = 0.5  # scalar truncation value in [0.02, 1.0]
# z = truncation * tf.random.truncated_normal([BATCH_SIZE, 128])  # noise sample
# y_index = tf.random.uniform([BATCH_SIZE], maxval=1000, dtype=tf.int32)
# y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label

# # Call BigGAN on a dict of the inputs to generate a batch of images with shape
# # [8, 512, 512, 3] and range [-1, 1].
# samples = module(dict(y=y, z=z, truncation=truncation))
# %%
#######GAN######################
# Define the generator network


# def make_generator_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=input_shape),
#         tf.keras.layers.Reshape((1, 1, input_shape[1])),
#         tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(output_factorized_0[0], output_factorized_1[0]), padding='same', activation=None),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.LeakyReLU(),
#         tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(output_factorized_0[1], output_factorized_1[1]), padding='same', activation=None),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.LeakyReLU(),
#         tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(output_factorized_0[2], output_factorized_1[2]), padding='same', activation=None),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.LeakyReLU(),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(np.prod(output_shape), activation='linear'),
#         tf.keras.layers.Reshape(output_shape)
#     ])
#     return model

# # Define the discriminator network
# def make_discriminator_model():
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Input(shape=output_shape))
#     model.add(tf.keras.layers.Reshape((output_shape[0], output_shape[1], 1)))
#     model.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(64,kernel_size=3,padding="same")))
#     model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
#     model.add(tf.keras.layers.MaxPool2D(pool_size =2, padding ='same'))
#     #model.add(Dropout(0.25))
#     model.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(64, kernel_size=3,padding="same")))
#     # model.add(tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))
#     model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
#     model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
#     return model

# # def make_generator_model():
# #     model = tf.keras.Sequential([
# #         tf.keras.layers.Input(shape=input_shape),
# #         tf.keras.layers.Reshape((1, 1, input_shape[1])),
# #         tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(output_factorized_0[0], output_factorized_1[0]), padding='same', activation=None),
# #         tf.keras.layers.LeakyReLU(),
# #         tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(output_factorized_0[1], output_factorized_1[1]), padding='same', activation=None),
# #         tf.keras.layers.LeakyReLU(),
# #         tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(output_factorized_0[2], output_factorized_1[2]), padding='same', activation=None),
# #         tf.keras.layers.LeakyReLU(),
# #         tf.keras.layers.Flatten(),
# #         tf.keras.layers.Dense(np.prod(output_shape), activation='linear'),
# #         tf.keras.layers.Reshape(output_shape)
# #     ])
# #     return model

# # def make_discriminator_model():
# #     model = tf.keras.Sequential()
# #     model.add(tf.keras.layers.Input(shape=output_shape))
# #     model.add(tf.keras.layers.Reshape((output_shape[0], output_shape[1], 1)))
# #     model.add(tf.keras.layers.Conv2D(64,kernel_size=3,padding="same"))
# #     model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
# #     model.add(tf.keras.layers.MaxPool2D(pool_size =2, padding ='same'))
# #     #model.add(Dropout(0.25))
# #     model.add(tf.keras.layers.Conv2D(64, kernel_size=3,padding="same"))
# #     model.add(tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))
# #     model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
# #     model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
# #     model.add(tf.keras.layers.Flatten())
# #     model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
# #     return model

# # Define the GAN model
# def make_gan_model(generator, discriminator):
#     discriminator.trainable = False
#     model = tf.keras.Sequential([generator, discriminator])
#     return model

# # Define the loss functions and optimizers
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)


# # Define the hyperparameters
# BUFFER_SIZE = 512
# BATCH_SIZE = 32
# EPOCHS = 2000
# # NOISE_DIM = input_shape[1]

# # Define the training loop
# # @tf.function
# # def train_step(profiles, images):
# #     # noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
# #     noise = profiles
    
# #     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
# #         generated_images = generator(noise, training=True)
        
# #         real_output = discriminator(images, training=True)
# #         fake_output = discriminator(generated_images, training=True)
        
# #         gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
# #         disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
        
# #     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
# #     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
# #     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
# #     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# def wasserstein_loss(y_true, y_pred):
#     return tf.reduce_mean(y_true * y_pred)

# # Define the number of training iterations and batch size
# num_iterations = 10
# # Define the training loop
# @tf.function
# def train_step(profiles, images):
#     # Sample random noise
#     noise = profiles
    
#     # Train the discriminator
#     # for i in range(num_iterations):
#     with tf.GradientTape() as discriminator_tape:
#         # Generate fake images
#         generated_images = generator(noise, training=True)
        
#         # Compute the discriminator's output for real and fake images
#         real_output = discriminator(images, training=True)
#         fake_output = discriminator(generated_images, training=True)
        
#         # Compute the discriminator's Wasserstein loss
#         discriminator_loss = wasserstein_loss(real_output, fake_output)
        
#     # Compute the gradients of the discriminator's trainable variables
#     discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    
#     # Apply the discriminator's optimizer to update its trainable variables
#     discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
#     # Clip the discriminator's weights to ensure Lipschitz continuity
#     for weight in discriminator.trainable_variables:
#         weight.assign(tf.clip_by_value(weight, -0.01, 0.01))

#     # Train the generator
#     for i in range(num_iterations):
#         with tf.GradientTape() as generator_tape:
#             # Generate fake images
#             generated_images = generator(noise, training=True)
            
#             # Compute the discriminator's output for the fake images
#             fake_output = discriminator(generated_images, training=True)
            
#             # Compute the generator's Wasserstein loss
#             generator_loss = -tf.reduce_mean(fake_output)
            
#         # Compute the gradients of the generator's trainable variables
#         generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
        
#         # Apply the generator's optimizer to update its trainable variables
#         generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        
#     return discriminator_loss, generator_loss

# # Define the training function
# def train(dataset, epochs):
#     discriminator_losses = []
#     generator_losses = []
#     for epoch in range(epochs):
#         print(f'Epoch: {epoch}')
#         for profiles, images in dataset.take(BATCH_SIZE):
#             discriminator_loss, generator_loss = train_step(profiles, images)

#         if epoch % 100 == 0:
#             discriminator_losses.append(discriminator_loss)
#             generator_losses.append(generator_loss)
#             images = generator(profiles[0:1], training=False)
#             plt.figure(figsize= (1,2))
#             plt.imshow(tf.transpose(images[0,:,:]))
#             plt.show()

# # Create the generator, discriminator, and GAN models
# generator = make_generator_model()
# discriminator = make_discriminator_model()
# gan = make_gan_model(generator, discriminator)

# # Create a tf.data.Dataset object from your input and target data
# # Use mixed precision training
# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# tf.keras.mixed_precision.experimental.set_policy(policy)

# # # Create a tf.data.Dataset object from your input and target data
# # dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# dataset = train_dataset
# # Train the GAN
# #%%
# train(dataset, EPOCHS)
# #%%
# num_plots = 4 
# rhos = np.linspace(0,1,input_shape[1])
# fig, axs = plt.subplots(figsize= (10,16),nrows=num_plots, ncols = 3, gridspec_kw={'width_ratios': [2, 1, 1]}, tight_layout =True)
# plt.suptitle(f"{process_str.capitalize()} X-Ray Image Predictions from Ion Temp Profile")
# i = 0
# for inputs, outputs in test_dataset.take(num_plots): 
#     images = generator(inputs, training=False)
#     axs[i][0].plot(rhos, inputs[0,0,:], label='input',color = 'orange')
#     axs[i][0].set_title(f'input profile')
#     axs[i][0].set_ylabel('temp (keV)')
#     axs[i][0].set_xlabel('rho')
#     axs[i][1].imshow(tf.transpose(outputs[0,:,:]))
#     axs[i][1].set_title(f'true image')
#     axs[i][1].set_ylabel('line of sight')
#     axs[i][1].set_xlabel('wavelength')
#     # axs[i][2].imshow(tf.transpose(images[0,:,:]))
#     axs[i][2].imshow(tf.transpose(images[0,:,:]))
#     # error = gan.evaluate(inputs,outputs, steps=1, batch_size=1)
#     # axs[i][2].set_title(f'prediction with MAE: {error:.1f}')
#     axs[i][2].set_ylabel('line of sight')
#     axs[i][2].set_xlabel('wavelength')
#     # axs[i][2].legend(loc='upper left',bbox_to_anchor=(0.8,1.24))
#     i += 1
# # %%
