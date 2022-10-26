# -*- coding: utf-8 -*-
"""
Mapping X-ray images to ion temperature profiles
Joel Aftreth, Max Planck Institute for Plasma Physics, Greifswald DE
based on @author: aktas
"""
# %% 
import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D , Dropout, MaxPool2D, \
    Flatten, Dense,  Activation, BatchNormalization, ZeroPadding2D, LeakyReLU, Lambda 
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2 , L1L2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.callbacks import CSVLogger
import h5py
import random 
import numpy as np
import seaborn as sns

# %% 
physical_devices = tf.config.list_physical_devices()
print("DEVICES : \n", physical_devices)


print('Using:')
print('\t\u2022 Python version:',sys.version)
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')



random.seed(123)
# %% 
GPU=True
if GPU:
    BATCH_SIZE = 200 # try 20, 50, 100, 64 seems to be max for GPU
    num_epochs = 500
    num_patience = 450
    steps = 10
    val_steps = 10
    num_schedule_reduce = int(10/steps)
    Normalize_Output = False
    learning_rate = 0.01 #decent results with 0.0001
    regularizer_pen = 0.0001
    dropout_rate = 0.5
    activ_fun = 'relu' #leaky_relu
    regularizer =  None #tf.keras.regularizers.OrthogonalRegularizer() #needs tf 2.10 #l2(regularizer_pen), L1L2(regularizer_pen,regularizer_pen)
    dense_regularizer = tf.keras.regularizers.OrthogonalRegularizer()
    kernel_size = (3,3) #3 originally
    stride = 1# (1,50) #(1,50) 
    pool_size = (2,2) #originally 2
    pool_stride = None #(1,50) #originally 2
    lr_patience = 2
    line_of_sight_start = 200
    cosine_importance = 1
    k_division = 1 # how many fold to divide dataset, and use most recent one
    huber_delta = 0.2
    loss_fn = tf.keras.losses.Huber(delta = huber_delta)
    # def joint_loss(y_true, y_pred):
    # return cosine_importance*tf.keras.losses.cosine_similarity(y_true, y_pred) + \
    #     tf.keras.losses.huber(y_true, y_pred, delta = huber_delta)
    Augment = False
    intensity_threshold = 0.5e5
    out_label_threshold = 5
    out_sigma_threshold =  2.5
    height_factor = 4 # num of pixels
    width_factor = 8 # num of pixels
    rotation_factor = 4/360 # degrees of circle
    load_pretrained = False
    high_temp_sample_weight = 2 # 1/10 profiles have > 2 keV

if GPU:
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%     
base_dir = r"C:\Users\joaf\Documents\models"
MODEL_FNAME = base_dir+r"\trained_model.h5"
tmp_model_name = base_dir+r"\tmp.h5"
save_plots_dir = r"C:\Users\joaf\Documents\results"



""" Prepare Input Data"""

def extract_mean_var(x, axes=0): # originally axes=[0,1]
    mean, variance = tf.nn.moments(x, axes=axes)
    return mean, variance
    
hdf5_path = r'C:\Users\joaf\Documents\Ion_Temp_Dataset.h5'
with h5py.File(hdf5_path, "r") as f:
    # Print all root level object names (aka keys) 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    train_input_key = list(f.keys())[3]
    train_output_key = list(f.keys())[4]
    train_output_sigma_key = list(f.keys())[5]
    valid_input_key = list(f.keys())[6]
    valid_output_key = list(f.keys())[7]
    valid_output_sigma_key = list(f.keys())[8]
    test_input_key = list(f.keys())[0]
    test_output_key = list(f.keys())[1]
    test_output_sigma_key = list(f.keys())[2]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[train_input_key])) 

    # preferred methods to get dataset values:
    num_train_imgs = f[train_input_key].shape[0]
    num_valid_imgs = f[valid_input_key].shape[0]
    num_test_imgs = f[test_input_key].shape[0]
    input_arr = f[train_input_key][:10]  # returns as a numpy array
    output_arr = f[train_output_key][:10]  # returns as a numpy array
    output_sigmas = f[train_output_sigma_key][:10] # tf.float32
    mean_train, var_train = extract_mean_var(f[train_output_key][:])
    mean_valid, var_valid = extract_mean_var(f[valid_output_key][:])
    mean_test, var_test = extract_mean_var(f[test_output_key][:])


def normalize_with_moments(x, mean, variance, epsilon=1e-8):    
    x_normed = (x - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return x_normed

def unnormalize_with_moments(x, mean, variance, epsilon=1e-8):    
    x_unnormed = x * tf.sqrt(variance + epsilon) + mean  # epsilon to avoid dividing by zero
    return x_unnormed


mean_total = (num_train_imgs*mean_train + num_valid_imgs*mean_valid + num_test_imgs*mean_test)/(num_train_imgs + num_valid_imgs + num_test_imgs)
var_total = (num_train_imgs*var_train + num_valid_imgs*var_valid + num_test_imgs*var_test)/(num_train_imgs + num_valid_imgs + num_test_imgs)
means_vars = {train_output_key:[mean_train, var_train],valid_output_key:[mean_valid, var_valid],test_output_key:[mean_test, var_test],\
    'total_output':[mean_total,var_total]}


# %%     
class generator:
    def __init__(self, file, input_key, output_key, output_sigma_key):
        self.file = file
        self.input_key = input_key
        self.output_key = output_key
        self.output_sigma_key = output_sigma_key

    def __call__(self):
        #makes interesting learning curves when taken in order instead of randomly
        with h5py.File(self.file, 'r') as hf:
            while True:
                if 'train' in self.input_key:
                    start_good_trials = int(num_train_imgs/k_division)
                elif 'valid' in self.input_key:
                    start_good_trials = int(num_valid_imgs/k_division)
                elif 'test' in self.input_key:
                    start_good_trials = int(num_test_imgs/k_division)    
                idx = random.randrange(start_good_trials)
                #idx = random.randrange(hf[self.input_key].shape[0])
                in_im = hf[self.input_key][-idx]
                in_im = in_im.reshape(input_arr.shape[1],input_arr.shape[2],1) #shape of (195,1475,1)
                in_im = in_im[:,line_of_sight_start:,:]
                
                intenstity = np.sum(in_im)
        
                out_im = hf[self.output_key][-idx]
                out_sigma = hf[self.output_sigma_key][-idx]
                
                out_im_max = np.max(out_im)
                out_sigma_max = np.max(out_sigma)
                
                if out_im_max>2:
                    sample_weight = high_temp_sample_weight
                else:
                    sample_weight = 1
                
                if Normalize_Output:
                    out_im = normalize_with_moments(out_im, means_vars['total_output'][0],means_vars['total_output'][1])
                # out_im = normalize_with_moments(out_im, means_vars[self.output_key][0],means_vars[self.output_key][1])
            
                
                if intenstity>intensity_threshold and out_im_max<out_label_threshold and out_sigma_max<out_sigma_threshold: ### keeps 75% of images
                    yield in_im, (out_im, out_sigma), sample_weight

image_shape = (tf.TensorShape([input_arr.shape[1],input_arr.shape[2]-line_of_sight_start,1]))


train_dataset = tf.data.Dataset.from_generator(
    generator(hdf5_path,train_input_key,train_output_key,train_output_sigma_key), 
    output_types = (tf.float32, (tf.float32, tf.float32), tf.int32),
    output_shapes=( image_shape, ((tf.TensorShape([output_arr.shape[1],])),(tf.TensorShape([output_arr.shape[1],]))),tf.TensorShape(None)))          
valid_dataset = tf.data.Dataset.from_generator(
    generator(hdf5_path,valid_input_key,valid_output_key,valid_output_sigma_key), 
    output_types = (tf.float32, (tf.float32, tf.float32),tf.int32),
    output_shapes=(image_shape, ((tf.TensorShape([output_arr.shape[1],])),(tf.TensorShape([output_arr.shape[1],]))),tf.TensorShape(None)))   
test_dataset = tf.data.Dataset.from_generator(
    generator(hdf5_path,test_input_key,test_output_key,test_output_sigma_key), 
    output_types = (tf.float32, (tf.float32, tf.float32),tf.int32),
    output_shapes=(image_shape , ((tf.TensorShape([output_arr.shape[1],])),(tf.TensorShape([output_arr.shape[1],]))),tf.TensorShape(None)))   


train_dataset = train_dataset.repeat().batch(BATCH_SIZE)

valid_dataset = valid_dataset.repeat().batch(BATCH_SIZE)

test_dataset = test_dataset.repeat().batch(BATCH_SIZE)






# %% 

""" Create Model"""

input = Input(shape =(input_arr.shape[1],input_arr.shape[2],1))

weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=123) #originally stddev was 0.01
bias_initializer=tf.keras.initializers.Zeros()

# Normalize input data to max of each individual image
max_val = tf.reduce_max(input)
x = tf.divide(input,max_val)
    
x = Conv2D (filters =32, kernel_size =(3,50), strides = (3,50), padding ='same', kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x)
# x = BatchNormalization()(x)
x = Activation(activ_fun)(x)
# x = MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same')(x)
# x = Dropout(dropout_rate)(x)
    
x = Conv2D (filters =32, kernel_size =kernel_size, strides = stride, padding ='same', kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x)
#x = BatchNormalization()(x)
x = Activation(activ_fun)(x)
x = MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same')(x)
# x = Conv2D (filters =64, kernel_size =kernel_size, strides = stride, padding ='same', kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x)
# x = BatchNormalization()(x)
# x = Activation(activ_fun)(x)
# # x = MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same')(x)
# # x = Dropout(dropout_rate)(x)


# x = Conv2D (filters =128, kernel_size =kernel_size, strides = stride, padding ='same', kernel_regularizer=regularizer,kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x)
# x = BatchNormalization()(x)
# x = Activation(activ_fun)(x)
# x = MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same')(x)



# x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# x = Dropout(0.5)(x)
# x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# x = Dropout(dropout_rate)(x)
x = Flatten()(x)
x = Dense(units = 128, activation =activ_fun, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 
x = Dense(units = 80, activation =activ_fun, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 
output = Dense(units = output_arr.shape[1], activation =None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 

# creating the model

model = Model (inputs=input, outputs =output)


#%%

def build_model(image_shape):
    weight_initializer = 'glorot_uniform'
    # weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=123)
    # use defaults
    model = Sequential()
    
    model.add(Input(shape=image_shape))
    def norm_image(input):
        normed = tf.image.per_image_standardization(input)
        return normed
    model.add(Lambda(norm_image))
    if Augment:
        model.add(tf.keras.layers.RandomTranslation(height_factor=height_factor/input_arr.shape[1],
            width_factor=width_factor/input_arr.shape[2],
            fill_mode='nearest',
            interpolation='nearest'))
        model.add(tf.keras.layers.RandomRotation(
            factor=rotation_factor,
            fill_mode='nearest',
            interpolation='nearest'))
    
    model.add(Conv2D(64, kernel_size=kernel_size, strides=stride, input_shape=image_shape,
                     padding="same",kernel_initializer=weight_initializer,kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same'))
    #model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=kernel_size, strides=stride, padding="same",\
        kernel_initializer=weight_initializer,kernel_regularizer=regularizer))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same'))
    #model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=kernel_size, strides=stride, padding="same",\
        kernel_initializer=weight_initializer,kernel_regularizer=regularizer))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same'))
    
    #model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=kernel_size, strides=stride, padding="same",kernel_regularizer=regularizer))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same'))

    # model.add(Dropout(0.25))
    # model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))

    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu',kernel_regularizer=dense_regularizer))
    model.add(BatchNormalization(momentum=0.8))
    # model.add(Dense(100, activation='relu',kernel_regularizer=dense_regularizer))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Dropout(0.25))
    model.add(Dense(output_arr.shape[1], activation="linear",kernel_regularizer=dense_regularizer))

    return model
model = build_model(image_shape)
#%%
# to be sure GPU memory is cleaned after last train
m = model
m.save(tmp_model_name)
del m
tf.keras.backend.clear_session()


model.summary()

""" Compile """


#compile the model by determining loss function Binary Cross Entropy, optimizer as SGD
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate,), #Adam originally
                loss= loss_fn, #joint_loss,
                metrics=[tf.keras.metrics.MeanSquaredError()],
                weighted_metrics=[],
                sample_weight_mode=[None])

#if validation accuracy doesnt improve for 15 epoch, stop training
early_stopping = EarlyStopping(monitor='val_loss', patience=num_patience,restore_best_weights=True)
    
#save the model if a better validation accuracy then previous better accuracy is obtained  
metric = 'val_loss'
checkpointer = ModelCheckpoint(filepath=MODEL_FNAME, verbose=2, monitor=metric, mode='min', save_best_only=True)
#schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,\
                              patience=lr_patience, min_lr=0.000001)
# write accuracy and loss history to the log.csv

log_dir = base_dir+r'\ilogs\default'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir) # in terminal run: tensorboard --logdir=C:\Users\joaf\Documents\models\ilogs\ --port=6006

csv_logger = CSVLogger(base_dir+r'\log.csv', append=True, separator=' ')

# %% 
""" Train """
fully_trained = False

if load_pretrained:
    model = tf.keras.models.load_model(MODEL_FNAME)#, custom_objects={'joint_loss': joint_loss})

history=model.fit(train_dataset,
    validation_data = valid_dataset,
    steps_per_epoch = steps,
    validation_steps = val_steps,
    epochs = num_epochs,
    verbose = 1,
    callbacks = [checkpointer,csv_logger,early_stopping]) #reduce_lr,schedule_callback, tensorboard_callback
fully_trained = True
# %% 
""" Plot the train and validation Loss """
test_pretrained = True
if test_pretrained:
    saved_model = 'C:\\Users\\joaf\\Documents\\models\\trained_model_weighted_high_temp.h5'
    model = tf.keras.models.load_model(saved_model)

if not test_pretrained:
    if not fully_trained:
        history = model.history
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss: MSE')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','valid'], loc='upper right')
    plt.savefig(save_plots_dir+r"\loss.png")
    plt.show(block=False)



""" Evaluate on test data """
scores = model.evaluate(test_dataset,steps=int(1000/BATCH_SIZE))
print(scores)
print("Plotting Examples")
num_plots = 4
num_scatter = 40
rhos = np.linspace(0,1,output_arr.shape[1])

#plots per prediction
if Normalize_Output:
    ppp = 3
    p = 1
else:
    ppp = 2
    p = 0
fig, axs = plt.subplots(figsize= (10,16*ppp),nrows=num_plots*ppp, sharex=False, tight_layout =True)
i=0
for images, (labels, sigmas), _ in test_dataset.take(num_plots): 
    pred = model.predict(images)
    axs[i*ppp].fill_between(rhos, labels[0] - sigmas[0], labels[0] + sigmas[0],
                 color='gray', alpha=0.5)
    axs[i*ppp].plot(rhos, labels[0], label='true',color='orange')
    axs[i*ppp].plot(rhos, pred[0],label='predicted',color = 'b')
    error = model.evaluate(images,labels, steps=1, batch_size=1)
    axs[i*ppp].set_title(f'model prediction with MSE: {error[1]:.4f}')
    axs[i*ppp].set_ylabel('temp keV')
    axs[i*ppp].set_xlabel('rho')
    axs[i*ppp].legend(loc='upper left',bbox_to_anchor=(0.8,1.24))
    

    
    if Normalize_Output:
        truth = unnormalize_with_moments(labels, means_vars['total_output'][0],means_vars['total_output'][1])
        pred = unnormalize_with_moments(model.predict(images), means_vars['total_output'][0],means_vars['total_output'][1])
        unnorm_error = tf.keras.losses.mean_squared_error(truth[0], pred[0])
        unnorm_error_overall = tf.keras.losses.mean_squared_error(truth, pred)
        axs[i*ppp+p].plot(rhos, pred[0],label='predicted')
        axs[i*ppp+p].plot(rhos, truth[0], label='true')
        axs[i*ppp+p].set_title(f'(unnormalized) model prediction with MSE: {unnorm_error:.4f}')
        axs[i*ppp+p].set_ylabel('temp keV')
        axs[i*ppp+p].set_xlabel('rho')
        axs[i*ppp+p].legend(loc='upper left',bbox_to_anchor=(0.8,1.24))
    
    axs[i*ppp+p+1].imshow(images[0,:,:,0])
    axs[i*ppp+p+1].set_title(f'Input Image')
    axs[i*ppp+p+1].set_ylabel('rho')
    axs[i*ppp+p+1].set_xlabel('line of sight')
    i+=1
fig.suptitle(f'Overall MSE: {scores[1]:.4f}')
fig.savefig(save_plots_dir+r"\predictions.png")
    

plt.figure()
if Normalize_Output:
    plt.plot(rhos, np.mean(np.square(truth - pred), axis=0))
else:
    plt.plot(rhos, np.mean(np.square(labels - pred), axis=0))
plt.xlabel('rho')
plt.ylabel('MSE keV')
plt.title(f'Loss (MSE) at each rho [Raw Outputs]')
plt.savefig(save_plots_dir+r"\loss_per_rho.png")
plt.show(block=False)

batches2take = 10
intenstities_all = np.zeros(BATCH_SIZE*batches2take)
errors_all = np.zeros(BATCH_SIZE*batches2take)
sample_weights_all = np.zeros(BATCH_SIZE*batches2take)
labels_all = np.zeros((BATCH_SIZE*batches2take,output_arr.shape[1]))
sigmas_all = np.zeros((BATCH_SIZE*batches2take,output_arr.shape[1]))
pred_all = np.zeros((BATCH_SIZE*batches2take,output_arr.shape[1]))


i = 0
for images, (labels, sigmas), sample_weights in test_dataset.take(10): 
    intenstities = np.sum(images,axis=(1,2,3))
    intenstities_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = intenstities
    labels_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = labels
    sigmas_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = sigmas
    sample_weights_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = sample_weights
    
    
    pred = model.predict(images)
    pred_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = pred
    if Normalize_Output:
        truth = unnormalize_with_moments(labels, means_vars['total_output'][0],means_vars['total_output'][1])
        error = np.mean(np.square(truth - pred), axis=1)
        errors_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = error
    else:
        error = np.mean(np.square(labels - pred), axis=1)
        errors_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = error
    #thresholded_idx = np.where(intenstities<0.1e9)[0]
    #thresholded_idx = np.where(error>0.2)[0]
    i += 1 

plt.figure()
plt.plot(rhos,labels_all[np.argmax(errors_all)], color = 'orange')
plt.fill_between(rhos, labels_all[np.argmax(errors_all)] - sigmas_all[np.argmax(errors_all)], labels_all[np.argmax(errors_all)] + sigmas_all[np.argmax(errors_all)],
                color='gray', alpha=0.5)
plt.plot(rhos,pred_all[np.argmax(errors_all)], color ='b')
plt.xlabel('rho')
plt.ylabel('MSE keV')
plt.title(f'Worst Loss (MSE) of Batch: {errors_all[np.argmax(errors_all)]:.4f}')

good_error_idx = np.where(errors_all<0.01)[0]
error_per_rho = np.mean(np.square(np.array(labels_all)[good_error_idx] - np.array(pred_all)[good_error_idx]), axis=0)
plt.figure()
plt.plot(rhos,error_per_rho)
plt.xlabel('rho')
plt.ylabel('MSE keV')
plt.title(f'Loss (MSE) at each rho [Good Outputs < 0.01 MSE]')
plt.savefig(save_plots_dir+r"\loss_per_rho_good_trials.png")
plt.show(block=False)

h = sns.jointplot(x=intenstities_all[:], y=errors_all[:],ratio=2,kind='hist') #,marginal_kws=dict(bins=20)
h.plot_joint(sns.kdeplot, color="r", zorder=0, levels=8)
h.plot_marginals(sns.rugplot, color="r", height=-.1, clip_on=False)
h.set_axis_labels('Intensity', 'MSE keV', fontsize=16)
plt.suptitle(f'Loss (MSE) vs Intensity [Raw Outputs]')
plt.tight_layout()
plt.savefig(save_plots_dir+r"\loss_vs_intensity.png")
plt.show(block=False)
    
# find experiment number for corresponding bad profiles   
num_bad=6    
with h5py.File(hdf5_path, "r") as f:
    test_output_key = list(f.keys())[4]
    sorted_max_labels = np.sort(np.max(f[test_output_key][:],axis=1))
    sorted_max_idx = np.argsort(np.max(f[test_output_key][:],axis=1))
    max_labels = f[test_output_key][np.sort(sorted_max_idx[-num_bad:-1])]
    mean_labels = np.mean(f[test_output_key][:], axis=0)
    #print(sorted_max_labels[-60:])
with h5py.File(hdf5_path, "r") as f:
    test_output_sigma_key = list(f.keys())[5]
    sorted_max_sigmas = np.sort(np.max(f[test_output_sigma_key][:],axis=1))
    max_sigmas = f[test_output_sigma_key][np.sort(sorted_max_idx[-num_bad:-1])]
    mean_sigmas = np.mean(f[test_output_sigma_key][:], axis=0)
    #print(sorted_max_sigmas[-60:])

plt.figure()
for i in range(len(max_labels)):
    plt.plot(rhos,max_labels[i]) 
    plt.fill_between(rhos, max_labels[i] - max_sigmas[i], max_labels[i] + max_sigmas[i],
                    color='gray', alpha=0.5)
plt.xlabel('rho')
plt.ylabel('Ion Temp keV')
plt.title(f'Worst Profiles [Raw Outputs]')
plt.show(block=False)

plt.figure()
if Normalize_Output:
    plt.plot(rhos, np.sqrt(np.mean(np.square(truth - pred), axis=0)), label = 'RMSE prediction')
else:
    plt.plot(rhos, np.sqrt(np.mean(np.square(labels_all - pred_all), axis=0)), label = 'RMSE prediction')
plt.plot(rhos, mean_sigmas, label= 'Sigma')
plt.plot(rhos,np.sqrt(error_per_rho), label='RMSE good predictions')
plt.xlabel('rho')
plt.ylabel('sigma from Novi keV')
plt.title(f'Sigma at each rho [Raw Outputs]')
plt.legend(loc='upper right')
plt.savefig(save_plots_dir+r"\sigmas_per_rho.png")
plt.show(block=False)

    
plt.figure() # for scatter plot
plt.axline([0, 0], [1, 1])
for i in range(min(num_scatter,BATCH_SIZE)):
    plt.scatter(model.predict(images)[i],labels[i],alpha=0.3)
plt.xlabel('Predicted keV')
plt.ylabel('True keV')
plt.suptitle(f'Scatter with Overall Loss: MSE (keV) {scores[1]:.4f}')
plt.tight_layout()
plt.savefig(save_plots_dir+r"\scatter.png")
plt.show(block=False)




    
    
    
print("End of Training")
# %% 

#tf.keras.backend.clear_session()
print('all done') 

# """ Learning Rate Scheduler """



# def scheduler(epoch, lr): #tensorflow suggestion
#     global count
#     count += 1
#     if epoch < 10:
#         return lr
#     else:
#         if count >= num_schedule_reduce:
#             count = 0
#             # return lr * tf.math.exp(-0.1)
#             return lr * 0.999
#         else:
#             return lr
        
# %%
