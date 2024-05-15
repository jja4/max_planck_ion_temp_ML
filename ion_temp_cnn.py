# -*- coding: utf-8 -*-
"""
Mapping X-ray images to ion temperature profiles
Joel Aftreth, Max Planck Institute for Plasma Physics, Greifswald DE
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
import gc
import w7xarchive

# %% 
"""Print Versions and GPU Devices"""
physical_devices = tf.config.list_physical_devices()
print("DEVICES : \n", physical_devices)
print('Using:')
print('\t\u2022 Python version:',sys.version)
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if len(tf.config.list_physical_devices('GPU'))>0 else '\t\u2022 GPU device not found. Running on CPU')

random.seed(123)
#%% 
"""
Hyperparameters ##############################
"""
GPU=True

"""model parameters"""
Transformer_On = False
Sequential_Model = True
regularizer_pen = 0.0001
dropout_rate = 0.5
activ_fun = 'gelu' #leaky_relu
regularizer =  None #tf.keras.regularizers.OrthogonalRegularizer() #needs tf 2.10 #l2(regularizer_pen), L1L2(regularizer_pen,regularizer_pen)
dense_regularizer = None #tf.keras.regularizers.OrthogonalRegularizer()
kernel_size = (3,3) #3 originally
stride = 1# (1,50) #(1,50) 
pool_size = (2,2) #originally 2
pool_stride = None #(1,50) #originally 2

"""training parameters"""
BATCH_SIZE = 100 # try 20, 50, 100, 64 seems to be max for GPU
num_epochs = 100
num_patience = int(num_epochs*0.75)
learning_rate = 0.001 #decent results with 0.0001
steps = 20
val_steps = 10
lr_patience = 2
k_division = 6 # how many fold to divide dataset, and use most recent one

"""loss function parameters"""
huber_delta = 0.2
loss_fn = tf.keras.losses.Huber(delta = huber_delta)
cosine_importance = 1

"""preprocessing parameters"""
Normalize_Output = False
Normalize_Input_Global = False
intensity_threshold = 0.5e5 # try with lower threshold 0.1e5
intensity_max = 0.3e6
#crop input image
Extra_Crop = True
if Extra_Crop:
    wavelength_start =105
    wavelength_end =180
    line_of_sight_start = 300
    line_of_sight_end = 875
    intensity_threshold = int(intensity_threshold/4) # try with lower threshold 0.1e5
    intensity_max = int(intensity_max/4)
out_label_threshold = 2.5 # switch to 3.5, how many above 2.5 in %?
out_sigma_threshold =  0.5

wavelength_start =105 # full image = 0
wavelength_end =185 # full image = 195
line_of_sight_start = 200 # full image = 0
line_of_sight_end = 1475 # full image = 1475

"""augmentation parameters"""
Augment = True
height_factor = 4 # num of pixels
width_factor = 8 # num of pixels
rotation_factor = 4/360 # degrees of circle
high_temp_sample_weight = 4 # 1/10 profiles have > 2 keV

"""ensemble parameters"""
Ensemble_Models = False
Ensemble_Simple_Average = False
Locally_Connected = False
Inverse_Selection_of_Dataset = False
Freeze_Ensemble_Models = True

Plot_Metrics = True
Test_Only = True
Collect_New_Data = False

if GPU:
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%%  
"""Where to load data and save trained model, plots"""
load_pretrained = True 
saved_model = 'C:\\Users\\joaf\\Documents\\models\\trained_model_extra_crop_fine_tuned_twice.h5'
hdf5_path = r'C:\Users\joaf\Documents\Ion_Temp_Dataset.h5'

base_dir = r"C:\Users\joaf\Documents\models"
MODEL_FNAME = base_dir+r"\trained_model_extra_crop_fine_tuned_twice_lower_intensity_threshold.h5"
save_plots_dir = r"C:\Users\joaf\Documents\results"


"""EDA ####################################"""
# used for normalizing data
def extract_mean_var(x, axes=0): # originally axes=[0,1]
    mean, variance = tf.nn.moments(x, axes=axes)
    return mean, variance
    
"""extract mean and variance of ion temp profiles to be plotted later"""
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

    num_train_imgs = f[train_input_key].shape[0]
    num_valid_imgs = f[valid_input_key].shape[0]
    num_test_imgs = f[test_input_key].shape[0]

    # example data
    input_arr = f[train_input_key][:10]  # returns as a numpy array
    output_arr = f[train_output_key][:10]  # returns as a numpy array
    output_sigmas = f[train_output_sigma_key][:10] # tf.float32
    
    if Normalize_Input_Global:
        mean_input, var_input = extract_mean_var(f[valid_input_key][:])
        mean_input = tf.reshape(mean_input,[mean_input.shape[0],mean_input.shape[1],1])
        var_input = tf.reshape(var_input,[var_input.shape[0],var_input.shape[1],1])
        mean_input = tf.cast(mean_input, tf.float32)
        var_input = tf.cast(var_input, tf.float32)
    mean_train, var_train = extract_mean_var(f[train_output_key][:])
    mean_valid, var_valid = extract_mean_var(f[valid_output_key][:])
    mean_test, var_test = extract_mean_var(f[test_output_key][:])
    min_train_idx = tf.argmin(f[train_output_key][:],axis=0)
    min_train = f[train_output_key][min_train_idx[0]]
    max_train_idx = tf.argmax(f[train_output_key][:],axis=0)
    max_train = f[train_output_key][max_train_idx[0]]
    med_train = np.median(f[train_output_key][:],axis=0)

"""plot an outlier datapoint and a "normal" datapoint"""
with h5py.File(hdf5_path, "r") as f: 
    below_plotted = False
    above_plotted = False
    while not above_plotted:   
        rand_img = random.randrange(num_train_imgs)
        in_im = f[train_input_key][rand_img]
        in_im = in_im.reshape(input_arr.shape[1],input_arr.shape[2],1) #shape of (195,1475,1)
        in_im = in_im[wavelength_start:wavelength_end,line_of_sight_start:line_of_sight_end,:]
        intensity = np.sum(in_im)
        if intensity_max > intensity > intensity_threshold:
            plt.figure()
            plt.imshow(tf.transpose(in_im[:,:,0]))
            plt.xlabel('wavelength')
            plt.ylabel('line of sight')   
            plt.title(f"Intensity of {intensity:.0f}")
            plt.savefig(save_plots_dir+r"\intensity_above_threshold.png")
            
            plt.figure()
            if Extra_Crop:
                plt.plot(in_im[:,500:510,0])
            else:
                plt.plot(in_im[:,600:610,0])
            plt.ylabel("intensity")
            plt.xlabel("wavelength")
            ylims = plt.gca().get_ylim()
            plt.title(f"Intensity of {intensity:.0f}: 10 Central Lines of Sight")
            plt.savefig(save_plots_dir+r"\intensity_above_threshold_10_lines.png")
            above_plotted = True
    while not below_plotted:   
        rand_img = random.randrange(num_train_imgs)
        in_im = f[train_input_key][rand_img]
        in_im = in_im.reshape(input_arr.shape[1],input_arr.shape[2],1) #shape of (195,1475,1)
        in_im = in_im[wavelength_start:wavelength_end,line_of_sight_start:line_of_sight_end,:]
        intensity = np.sum(in_im)
        if intensity < intensity_threshold:
            plt.figure()
            plt.imshow(tf.transpose(in_im[:,:,0]))
            plt.xlabel('wavelength')
            plt.ylabel('line of sight')   
            plt.title(f"Intensity of {intensity:.0f}")
            plt.savefig(save_plots_dir+r"\intensity_below_threshold.png")
            plt.figure()
            if Extra_Crop:
                plt.plot(in_im[:,500:510,0])
            else:
                plt.plot(in_im[:,600:610,0])
            plt.ylabel("intensity")
            plt.xlabel("wavelength")
            plt.gca().set_ylim(ylims)
            plt.title(f"Intensity of {intensity:.0f}: 10 Central Lines of Sight")
            plt.savefig(save_plots_dir+r"\intensity_below_threshold_10_lines.png")
            below_plotted = True
#%%
def normalize_with_moments(x, mean, variance, epsilon=1e-8):    
    x_normed = (x - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return x_normed

def unnormalize_with_moments(x, mean, variance, epsilon=1e-8):    
    x_unnormed = x * tf.sqrt(variance + epsilon) + mean  # epsilon to avoid dividing by zero
    return x_unnormed

"""calculate mean and variance of whole dataset"""
mean_total = (num_train_imgs*mean_train + num_valid_imgs*mean_valid + num_test_imgs*mean_test)/(num_train_imgs + num_valid_imgs + num_test_imgs)
var_total = (num_train_imgs*var_train + num_valid_imgs*var_valid + num_test_imgs*var_test)/(num_train_imgs + num_valid_imgs + num_test_imgs)
means_vars = {train_output_key:[mean_train, var_train],valid_output_key:[mean_valid, var_valid],test_output_key:[mean_test, var_test],\
    'total_output':[mean_total,var_total]}


# %%     
""" Prepare Input Data ####################################"""
"""Create generator to pull data from HDF5 file"""
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
                """randomly sample data from specified dataset"""
                if 'train' in self.input_key:
                    start_good_trials = int(num_train_imgs/k_division)
                elif 'valid' in self.input_key:
                    start_good_trials = int(num_valid_imgs/k_division)
                elif 'test' in self.input_key:
                    start_good_trials = int(num_test_imgs/k_division)    
                idx = random.randrange(start_good_trials)
                in_im = hf[self.input_key][-idx]
                in_im = in_im.reshape(input_arr.shape[1],input_arr.shape[2],1) #shape of (195,1475,1)
                in_im = in_im[wavelength_start:wavelength_end,line_of_sight_start:line_of_sight_end,:]
                
                intensity = np.sum(in_im)
        
                out_im = hf[self.output_key][-idx]
                out_sigma = hf[self.output_sigma_key][-idx]
                
                out_im_max = np.max(out_im)
                out_sigma_max = np.max(out_sigma)
                
                """weight the samples, rarer samples get higher weighting"""
                if out_im_max>2.25 or out_im_max<1.52:
                    sample_weight = high_temp_sample_weight
                else:
                    sample_weight = 1
                
                if Normalize_Output:
                    out_im = normalize_with_moments(out_im, means_vars['total_output'][0],means_vars['total_output'][1])            
                if Inverse_Selection_of_Dataset:
                    if intensity<intensity_threshold or out_im_max>out_label_threshold or out_sigma_max>out_sigma_threshold: 
                        yield in_im, (out_im, out_sigma), sample_weight
                else:
                    """filter out outliers"""
                    if intensity_max>intensity>intensity_threshold and out_im_max<out_label_threshold and out_sigma_max<out_sigma_threshold: ### keeps 75% of images
                        yield in_im, (out_im, out_sigma), sample_weight
"""size of input images"""
image_shape = (tf.TensorShape([wavelength_end-wavelength_start,line_of_sight_end-line_of_sight_start,1]))

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






#%%
""" Create Model"""
if Sequential_Model:
    def build_model(image_shape):
        weight_initializer = 'glorot_uniform'
        model = Sequential()
        
        model.add(Input(shape=image_shape))
        def norm_image(input):
            normed = tf.image.per_image_standardization(input)
            return normed
        model.add(Lambda(norm_image))
        if Normalize_Input_Global:
            def standardize_image_per_pixel(input):
                mean, variance, epsilon = mean_input[wavelength_start:wavelength_end,line_of_sight_start:line_of_sight_end,:], var_input[wavelength_start:wavelength_end,line_of_sight_start:line_of_sight_end,:], 1e-8    
                x_normed = (input - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
                return x_normed
            model.add(Lambda(standardize_image_per_pixel))
        
        if Augment:
            model.add(tf.keras.layers.RandomTranslation(height_factor=height_factor/(wavelength_end-wavelength_start),
                width_factor=width_factor/(line_of_sight_end-line_of_sight_start),
                fill_mode='nearest',
                interpolation='nearest'))
            model.add(tf.keras.layers.RandomRotation(
                factor=rotation_factor,
                fill_mode='nearest',
                interpolation='nearest'))
        if Locally_Connected:
            model.add(tf.keras.layers.LocallyConnected2D(1,kernel_size=(80,int(1275/40)),strides=(80,int(1275/40)),padding='same',implementation=2,activation = 'relu'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Flatten())
            model.add(Dense(output_arr.shape[1], activation="linear",kernel_regularizer=dense_regularizer))
            return model
        else:
            model.add(Conv2D(64, kernel_size=kernel_size, strides=stride, input_shape=image_shape,
                            padding="same",kernel_initializer=weight_initializer,kernel_regularizer=regularizer))
            model.add(LeakyReLU(alpha=0.2))
            model.add(MaxPool2D(pool_size =pool_size, strides =2, padding ='same'))
            #model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=kernel_size, strides=stride, padding="same",\
                kernel_initializer=weight_initializer,kernel_regularizer=regularizer))
            model.add(ZeroPadding2D(padding=((0,1),(0,1))))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))

            model.add(MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same'))
            #model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=kernel_size, strides=stride, padding="same",\
                kernel_initializer=weight_initializer,kernel_regularizer=regularizer))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same'))
            
            # model.add(Dropout(0.25))
            model.add(Conv2D(32, kernel_size=kernel_size, strides=stride, padding="same",kernel_regularizer=regularizer))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same'))

            model.add(Flatten())
            model.add(Dense(output_arr.shape[1], activation="linear",kernel_regularizer=dense_regularizer))

            return model
    
    model = build_model(image_shape)
#%%
"""Load trained models into an ensemble models if Ensemble_Models = True"""
def load_all_models():
    all_models = []
    model_names = [f'trained_model_{i}.h5' for i in np.arange(0,20)]
    for model_name in model_names:
        filename = os.path.join(base_dir, model_name)
        model = tf.keras.models.load_model(filename)
        all_models.append(model)
        print('loaded:', filename)
    return all_models

def ensemble_model(models,Ensemble_Simple_Average):
    for i, model in enumerate(models):
        model._name = f'{model._name}_{i}' 
        if Freeze_Ensemble_Models:
            for layer in model.layers:
                layer.trainable = False
    model_input = tf.keras.Input(shape=image_shape)
    
    ensemble_outputs = [model(model_input) for model in models]
    print(model(model_input).shape)
    
    if Ensemble_Simple_Average:
        output = tf.keras.layers.Average()(ensemble_outputs)
    else:
        merge = tf.keras.layers.concatenate(ensemble_outputs, axis = 1)
        merge = tf.keras.layers.Reshape((output_arr.shape[1],len(models)))(merge) 
        local_2d = tf.keras.layers.LocallyConnected1D(1,kernel_size=len(models),padding='same',implementation=2,activation = 'linear')(merge)
        output = tf.keras.layers.Flatten()(local_2d)
        output = tf.keras.layers.Dense(40,activation = 'linear')(output)
    model = tf.keras.models.Model(inputs=model_input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.Huber(huber_delta), metrics=[tf.keras.metrics.MeanSquaredError()],weighted_metrics=[])
    return model

if Ensemble_Models:
    models = load_all_models()
    model = ensemble_model(models,Ensemble_Simple_Average)
    MODEL_FNAME = base_dir+f"\\trained_model_ensemble_{len(models)}.h5"

"""load a pre-trained model instead of training from scratch"""
if load_pretrained:
    model = tf.keras.models.load_model(saved_model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate,), #Adam originally
                        loss= loss_fn, #joint_loss,
                        metrics=[tf.keras.metrics.MeanSquaredError()],
                        weighted_metrics=[],
                        sample_weight_mode=[None])

"""show model parameter summary"""
model.summary()

""" 
Compile the model: 
    determine the loss function and optimizer
"""
if not Test_Only:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate,), #Adam originally
                    loss= loss_fn, #joint_loss,
                    metrics=[tf.keras.metrics.MeanSquaredError()],
                    weighted_metrics=[],
                    sample_weight_mode=[None])

"""if validation accuracy doesnt improve for 15 epoch, stop training"""
early_stopping = EarlyStopping(monitor='val_loss', patience=num_patience,restore_best_weights=True)
    
"""save model when validation loss declines"""
metric = 'val_loss'
checkpointer = ModelCheckpoint(filepath=MODEL_FNAME, verbose=2, monitor=metric, mode='min', save_best_only=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,\
                            patience=lr_patience, min_lr=0.000001)

"""write accuracy and loss history to the log.csv"""
log_dir = base_dir+r'\ilogs\default'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir) # in terminal run: tensorboard --logdir=C:\Users\joaf\Documents\models\ilogs\ --port=6006
csv_logger = CSVLogger(base_dir+r'\log.csv', append=True, separator=' ')

"""monitor which epoch is best"""
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
# %% 
""" Train """
fully_trained = False
if not Test_Only:
    history=model.fit(train_dataset,
        validation_data = valid_dataset,
        steps_per_epoch = steps,
        validation_steps = val_steps,
        epochs = num_epochs,
        verbose = 1,
        callbacks = [checkpointer,csv_logger,early_stopping,BestEpochCallback(), ClearMemory()]) 
    fully_trained = True

# %% 
"""Evaluate on Test Data and Plot Results ##################################"""
if Plot_Metrics:
    """ Plot the train and validation Loss """
    if not Test_Only:
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
    errors_1000 = np.zeros(BATCH_SIZE*int(1000/BATCH_SIZE))
    naive_errors_1000 = np.zeros(BATCH_SIZE*int(1000/BATCH_SIZE))
    errors_per_rho_1000 = np.zeros((BATCH_SIZE*int(1000/BATCH_SIZE),output_arr.shape[1]))
    if Normalize_Output:
        i =0
        for images, (labels, sigmas), _ in test_dataset.take(int(1000/BATCH_SIZE)): 
            truth = unnormalize_with_moments(labels, means_vars['total_output'][0],means_vars['total_output'][1])
            pred = unnormalize_with_moments(model.predict(images), means_vars['total_output'][0],means_vars['total_output'][1])
            errors_per_rho_1000[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = np.mean(np.square(truth - pred), axis=0)
            errors_1000[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = tf.keras.losses.mean_squared_error(truth, pred)
            i+=1
    else: 
        i =0
        for images, (labels, sigmas), _ in test_dataset.take(int(1000/BATCH_SIZE)): 
            truth = labels
            pred = model.predict(images)
            naive_pred = mean_train
            errors_per_rho_1000[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = np.mean(np.square(truth - pred), axis=0)
            errors_1000[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = tf.keras.losses.mean_squared_error(truth, pred)
            naive_errors_1000[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = tf.keras.losses.mean_squared_error(truth, naive_pred)

            i+=1
        scores = errors_1000.mean()
        naive_scores = naive_errors_1000.mean()
    print(f'Prediction Error {scores}')
    print(f'Naive Error {naive_scores}')
    print("Plotting Examples")
    num_plots = 4
    num_scatter = 40
    rhos = np.linspace(0,1,output_arr.shape[1])

    """Plots of random predictions vs true values"""
    if Normalize_Output:
        ppp = 3
        p = 1
    else:
        ppp = 2
        p = 0
    fig, axs = plt.subplots(figsize= (10,16*ppp),nrows=num_plots, ncols = 2, gridspec_kw={'width_ratios': [1, 5]}, tight_layout =True)

    i=0
    for images, (labels, sigmas), _ in test_dataset.take(num_plots): 
        pred = model.predict(images)
        axs[i][1].fill_between(rhos, labels[0] - sigmas[0], labels[0] + sigmas[0],
                    color='gray', alpha=0.5)
        axs[i][1].plot(rhos, labels[0], label='true',color='orange')
        axs[i][1].plot(rhos, pred[0],label='predicted',color = 'b')
        error = model.evaluate(images,labels, steps=1, batch_size=1)
        axs[i][1].set_title(f'model prediction with MSE: {error[1]:.4f}')
        axs[i][1].set_ylabel('temp keV')
        axs[i][1].set_xlabel('rho')
        axs[i][1].legend(loc='upper left',bbox_to_anchor=(0.8,1.24))
        

        
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
        
        axs[i][0].imshow(tf.transpose(images[0,:,:,0]))
        axs[i][0].set_title(f'Input Image')
        axs[i][0].set_xlabel('wavelength')
        axs[i][0].set_ylabel('line of sight')
        i+=1
    fig.suptitle(f'Overall MSE: {scores:.4f}')
    fig.savefig(save_plots_dir+r"\predictions.png")
    
    """Plot Example X-ray Input Image"""
    plt.figure(figsize= (12,8))
    plt.imshow(tf.transpose(images[0,:,:,0]))
    plt.title(f'Example X-Ray Image')
    plt.xlabel('wavelength')
    plt.ylabel('line of sight')    

    """Plot Loss per rho (plasma radius)"""
    plt.figure()
    plt.plot(rhos, np.mean(errors_per_rho_1000, axis=0))
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
        sigmas_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = sigmas
        sample_weights_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = sample_weights
    
        if Normalize_Output:
            pred = unnormalize_with_moments(model.predict(images), means_vars['total_output'][0],means_vars['total_output'][1])
            pred_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = pred
            truth = unnormalize_with_moments(labels, means_vars['total_output'][0],means_vars['total_output'][1])
            labels_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = truth
            error = np.mean(np.square(truth - pred), axis=1)
            errors_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = error
        else:
            pred = model.predict(images)
            pred_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = pred
            labels_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE),:] = labels
            error = np.mean(np.square(labels - pred), axis=1)
            errors_all[int(i*BATCH_SIZE):int((i+1)*BATCH_SIZE)] = error
        i += 1 

    """Plot poor performing prediction"""
    plt.figure()
    plt.plot(rhos,labels_all[np.argmax(errors_all)], color = 'orange')
    if not Normalize_Output:
        plt.fill_between(rhos, labels_all[np.argmax(errors_all)] - sigmas_all[np.argmax(errors_all)], labels_all[np.argmax(errors_all)] + sigmas_all[np.argmax(errors_all)],
                    color='gray', alpha=0.5)
    plt.plot(rhos,pred_all[np.argmax(errors_all)], color ='b')
    plt.xlabel('rho')
    plt.ylabel('MSE keV')
    plt.title(f'Worst Loss (MSE) of Batch: {errors_all[np.argmax(errors_all)]:.4f}')

    good_error_idx = np.where(errors_1000<0.001)[0]
    good_error_per_rho = np.mean(errors_per_rho_1000[good_error_idx,:], axis=0)


    """Plot joint histogram of Intensity (brightness) vs Loss """
    h = sns.jointplot(x=intenstities_all[:], y=errors_all[:],ratio=5,kind='hist') #,marginal_kws=dict(bins=20)
    h.plot_marginals(sns.rugplot, color="r", height=-.1, clip_on=False)
    h.set_axis_labels('Intensity', 'MSE keV', fontsize=16)
    plt.suptitle(f'Loss (MSE) vs Intensity [Raw Outputs]')
    plt.tight_layout()
    plt.savefig(save_plots_dir+r"\loss_vs_intensity.png")
    plt.show(block=False)

    """Plot joint histogram of Max Temp vs Loss """
    h = sns.jointplot(x=np.max(labels_all,axis=1), y=errors_all[:],ratio=5,kind='hist') #,marginal_kws=dict(bins=20)
    h.plot_marginals(sns.rugplot, color="r", height=-.1, clip_on=False)
    h.set_axis_labels('Max Temp keV', 'MSE keV', fontsize=16)
    plt.suptitle(f'Loss (MSE) vs Max Temp [Raw Outputs]')
    plt.tight_layout()
    plt.savefig(save_plots_dir+r"\loss_vs_temp.png")
    plt.show(block=False)
        
    """Plot random predictions to see how frequently bad predictions occur"""  
    num_bad=6    
    with h5py.File(hdf5_path, "r") as f:
        test_output_key = list(f.keys())[4]
        sorted_max_labels = np.sort(np.max(f[test_output_key][:],axis=1))
        print(sorted_max_labels.shape)
        sorted_max_idx = np.argsort(np.max(f[test_output_key][:],axis=1))
        max_labels = f[test_output_key][np.sort(sorted_max_idx[-num_bad:])[:-1]]
        mean_labels = np.mean(f[test_output_key][:], axis=0)

        plt.figure()
        num_plots = len(f[test_output_key][::10])
        n_rhos = np.repeat([rhos],num_plots,axis=0)
        plt.plot(n_rhos.T,f[test_output_key][::10].T,alpha = 0.1)
        plt.xlabel('rho')
        plt.ylabel('Ion Temp keV')
        plt.title('Every 10 Ion Temp Profiles')
        plt.savefig(save_plots_dir+r"\every_10_profile_plots.png")
        
    with h5py.File(hdf5_path, "r") as f:
        test_output_sigma_key = list(f.keys())[5]
        sorted_max_sigmas = np.sort(np.max(f[test_output_sigma_key][:],axis=1))
        max_sigmas = f[test_output_sigma_key][np.sort(sorted_max_idx[-num_bad:-1])]
        mean_sigmas = np.mean(f[test_output_sigma_key][:], axis=0)

    """Plot Mean, Median, Min, and Max Temp Profiles"""
    plt.figure()
    plt.plot(rhos,mean_train, label = 'mean')
    plt.plot(rhos,med_train, label = 'median')
    plt.plot(rhos,min_train, label = 'min')
    plt.plot(rhos,max_labels[-2], label = 'max') 
    plt.fill_between(rhos, mean_train - var_train, mean_train + var_train,
                        color='gray', alpha=0.5)
    plt.xlabel('rho')
    plt.ylabel('Ion Temp keV')
    plt.title(f'Ion Temperature Profile Metrics')
    plt.legend()
    plt.savefig(save_plots_dir+r"\ion_profile_metrics_plot.png")
    plt.show(block=False)

    """Plot Error per rho of predictions and Physics Model Error"""
    plt.figure()
    plt.plot(rhos, mean_sigmas, label= 'Sigma', color='orange')
    plt.plot(rhos, np.sqrt(np.mean(errors_per_rho_1000, axis=0)), label = 'RMSE predictions', color='b')
    plt.plot(rhos,np.sqrt(good_error_per_rho), label='RMSE<0.001 predictions', color='g')
    plt.xlabel('rho')
    plt.ylabel('Sigma from Novi keV')
    plt.title(f'Sigma at each rho [Raw Outputs]')
    plt.legend(loc='upper right')
    plt.savefig(save_plots_dir+r"\sigmas_per_rho.png")
    plt.show(block=False)

    """Plot Scatter plot of Predicted vs True"""    
    plt.figure() 
    plt.axline([0, 0], [-1, 1])
    for i in range(min(num_scatter,BATCH_SIZE)):
        if Normalize_Output:
            pred = unnormalize_with_moments(model.predict(images)[i], means_vars['total_output'][0],means_vars['total_output'][1])
            truth = unnormalize_with_moments(labels[i], means_vars['total_output'][0],means_vars['total_output'][1])
            error = np.mean(np.square(truth - pred))
            plt.scatter(-pred,truth,alpha=0.3)
        else:
            plt.scatter(-model.predict(images)[i],labels[i],alpha=0.3)
    locs, xlabels = plt.xticks()
    for i in range(len(xlabels)):
        xlabels[i].set_text(str(-1*float(xlabels[i].get_text().replace("−", "-"))))
    plt.xticks(locs,xlabels)
    plt.xlabel('Predicted keV')
    plt.ylabel('True keV')
    plt.suptitle(f'Scatter with Overall Loss: MSE (keV) {errors_1000.mean():.4f}')
    plt.tight_layout()
    plt.savefig(save_plots_dir+r"\scatter.png")
    plt.show(block=False)

    """Plot Random Predictions as Lines"""
    plt.figure() 
    for i in range(min(num_scatter,BATCH_SIZE)):
        if Normalize_Output:
            pred = unnormalize_with_moments(model.predict(images)[i], means_vars['total_output'][0],means_vars['total_output'][1])
            plt.plot(rhos, pred)
        else:
            plt.plot(rhos,model.predict(images)[i])
    plt.xlabel('rho')
    plt.ylabel('Predicted keV')
    plt.suptitle(f'Random Trial Predictions with Overall Loss: MSE (keV) {errors_1000.mean():.4f}')
    plt.tight_layout()
    plt.savefig(save_plots_dir+r"\pred_examples.png")
    plt.show(block=False)      

print("End of Training")





# %% 
"""Test model on newly acquired data (not in Test Dataset)"""
signal_name = "ArchiveDB/raw/W7X/ControlStation.71501/DETECTOR0-1_DATASTREAM/0/frames"
shotnum = '20221109.15'
time_intervals = w7xarchive.get_time_intervals_for_program(signal_name, shotnum)
print(time_intervals.shape)

# # reads the timestamp (in nanoseconds) corresponding to the beginning of the discharge. we will download data from this point in time
# from_time = w7xarchive.get_program_t1("20221109.015")
# # we will read data until this time point. this is just the starting point + 1 second (converted to nanosecond, hence 1*10^9).


def moving_average(a, n=10) :
    ret = np.cumsum(a, axis =0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def moving_starts(a,n=10):
    return a[n-1:]
"""Source new X-ray data"""
if Collect_New_Data:
    all_new_images =[]
    times = []
    good_times = []

    for tim_e in time_intervals[:]: #index 757 error
        from_time = tim_e[0]
        to_time = tim_e[1]
        t, d = w7xarchive.get_image_json("ArchiveDB/raw/W7X/ControlStation.71501/DETECTOR0-1_DATASTREAM/0/frames", from_time, to_time)
        times.append(t)

        """Apply the same preprocessing to the new data"""
        if len(d.shape)>2:
            shot_avgs = moving_average(d)
            start_ts =moving_starts(t)

            shot_avgs = shot_avgs[:,wavelength_start:wavelength_end,line_of_sight_start:line_of_sight_end]
                            
            new_images = shot_avgs.reshape(shot_avgs.shape[0],shot_avgs.shape[1],shot_avgs.shape[2],1)
            new_images[new_images>6e4]=1
            intensities = np.sum(new_images,axis=(1,2,3))
            good_new_images = np.where(intensities>intensity_threshold)
            if new_images[good_new_images].shape[0]>0:
                all_new_images.append(new_images[good_new_images])
                good_times.append(start_ts[good_new_images])
    new_good_times = np.concatenate(good_times)            
    new_times = np.concatenate(times)
    new_exp_data = np.concatenate(all_new_images)
    np.save(r'C:\Users\joaf\Documents\New_XICS_Image_times_091122_cropped.npy',new_good_times)
    np.save(r'C:\Users\joaf\Documents\New_XICS_Images_091122_cropped.npy',new_exp_data)
"""Load new data already preprocessed instead of sourcing it"""
if not Collect_New_Data:
    new_exp_data = np.load(r'C:\Users\joaf\Documents\New_XICS_Images_091122.npy')
    new_good_times = np.load(r'C:\Users\joaf\Documents\New_XICS_Image_times_091122.npy')
#%% New Ti Profiles
"""Source new corresponding ion temp data and Fast Physics Model Predictions"""
ti_signal_name = "ArchiveDB/raw/Minerva/Minerva.IonTemperature.XICS/Ti_lineIntegrated_DATASTREAM/V1/0/signalTi/"
time_from, time_to = time_intervals[-1,0], time_intervals[0,1]
time, val = w7xarchive.get_signal(ti_signal_name, time_from, time_to)

overlapping_times = np.intersect1d(new_good_times,time)
num_plots = 10
x_axis = np.repeat([np.arange(0,val.shape[1])],num_plots,axis=0)
plt.plot(x_axis.T, val[:10].T)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], value
#%% Plots
"""Plot New CNN Predictions vs Fast Physics Model Predictions"""
model = tf.keras.models.load_model(MODEL_FNAME)
num_plots = 4
fig, axs = plt.subplots(figsize= (10,16*ppp),nrows=num_plots, ncols = 2, gridspec_kw={'width_ratios': [1, 5]}, tight_layout =True)

i = 0 
while True:
    rand = random.randrange(9,len(overlapping_times))
    if time[rand]-time[rand-10] == 1000000000:
        print(i)
        if i==num_plots:
            break
        axs[i][0].imshow(tf.transpose(new_exp_data[rand,:,:,0]))
        axs[i][0].set_title(f'Input Image')
        axs[i][0].set_xlabel('wavelength')
        axs[i][0].set_ylabel('line of sight')
        
        pred = model.predict(new_exp_data[rand:rand+1])
        rhos = np.linspace(0,1,output_arr.shape[1])
        axs[i][1].plot(rhos, pred[0],label='NN predicted',color = 'b')
        #truncate negative part of profile
        axs[i][1].plot(np.linspace(0,.91,len(val[rand])), np.mean(val[rand-9:rand+1].T-0.500,axis=1),label='bayesian',color = 'orange')
        new_line = '\n'
        axs[i][1].set_title(f'model prediction on new data {shotnum}{new_line}{time[rand-10]:.0f}:{time[rand]:.0f}')
        axs[i][1].set_ylabel('temp keV')
        axs[i][1].set_xlabel('rho')
        axs[i][1].legend(loc='upper left',bbox_to_anchor=(0.8,1.24))
        
        plt.savefig(save_plots_dir+r"\new_data_predictions.png")

        i+=1
            

plt.savefig(save_plots_dir+r"\new_data_predictions.png")
print('all done') 
# %%

