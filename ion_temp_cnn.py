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
import gc

# %% 
physical_devices = tf.config.list_physical_devices()
print("DEVICES : \n", physical_devices)


print('Using:')
print('\t\u2022 Python version:',sys.version)
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if len(tf.config.list_physical_devices('GPU'))>0 else '\t\u2022 GPU device not found. Running on CPU')



random.seed(123)
# %% 
GPU=True
if GPU:
    Transformer_On = False
    Sequential_Model = True
    Collect_New_Data = False
    BATCH_SIZE = 100 # try 20, 50, 100, 64 seems to be max for GPU
    num_epochs = 100
    num_patience = int(num_epochs*0.75)
    steps = 20
    val_steps = 10
    # num_schedule_reduce = int(10/steps)
    Normalize_Output = False
    Normalize_Input_Global = False
    learning_rate = 0.001 #decent results with 0.0001
    regularizer_pen = 0.0001
    dropout_rate = 0.5
    activ_fun = 'gelu' #leaky_relu
    regularizer =  None #tf.keras.regularizers.OrthogonalRegularizer() #needs tf 2.10 #l2(regularizer_pen), L1L2(regularizer_pen,regularizer_pen)
    dense_regularizer = None #tf.keras.regularizers.OrthogonalRegularizer()
    kernel_size = (3,3) #3 originally
    stride = 1# (1,50) #(1,50) 
    pool_size = (2,2) #originally 2
    pool_stride = None #(1,50) #originally 2
    lr_patience = 2
    wavelength_start =105 # full image = 0
    wavelength_end =185 # full image = 195
    line_of_sight_start = 200 # full image = 0
    line_of_sight_end = 1475 # full image = 1475
    
    #crop image
    Extra_Crop = True
    if Extra_Crop:
        wavelength_start =105
        wavelength_end =180
        line_of_sight_start = 300
        line_of_sight_end = 875
    
    k_division = 6 # how many fold to divide dataset, and use most recent one
    huber_delta = 0.2
    loss_fn = tf.keras.losses.Huber(delta = huber_delta)
    cosine_importance = 1
    # def joint_loss(y_true, y_pred):
    # return cosine_importance*tf.keras.losses.cosine_similarity(y_true, y_pred) + \
    #     tf.keras.losses.huber(y_true, y_pred, delta = huber_delta)
    Augment = True
    intensity_threshold = 0.5e5 # try with lower threshold 0.1e5
    intensity_max = 0.3e6
    if Extra_Crop:
        intensity_threshold = int(intensity_threshold/4) # try with lower threshold 0.1e5
        intensity_max = int(intensity_max/4)
    out_label_threshold = 2.5 # switch to 3.5, how many above 2.5 in %?
    out_sigma_threshold =  0.5
    height_factor = 4 # num of pixels
    width_factor = 8 # num of pixels
    rotation_factor = 4/360 # degrees of circle
    
    load_pretrained = True 
    saved_model = 'C:\\Users\\joaf\\Documents\\models\\trained_model_extra_crop_fine_tuned_twice.h5'
    # saved_model = 'C:\\Users\\joaf\\Documents\\models\\trained_model_mse_0038.h5'
    hdf5_path = r'C:\Users\joaf\Documents\Ion_Temp_Dataset.h5'
    #hdf5_path = r'C:\Users\joaf\Documents\Ion_Temp_Dataset_adj.h5'
    high_temp_sample_weight = 4 # 1/10 profiles have > 2 keV
    Ensemble_Models = False
    
    Test_Only = True
    Ensemble_Simple_Average = False
    Locally_Connected = False
    Inverse_Selection_of_Dataset = False
    Plot_Metrics = True
    Freeze_Ensemble_Models = True

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
MODEL_FNAME = base_dir+r"\trained_model_extra_crop_fine_tuned_twice_lower_intensity_threshold.h5"
save_plots_dir = r"C:\Users\joaf\Documents\results"



""" Prepare Input Data"""

def extract_mean_var(x, axes=0): # originally axes=[0,1]
    mean, variance = tf.nn.moments(x, axes=axes)
    return mean, variance
    

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
                in_im = in_im[wavelength_start:wavelength_end,line_of_sight_start:line_of_sight_end,:]
                
                intensity = np.sum(in_im)
        
                out_im = hf[self.output_key][-idx]
                out_sigma = hf[self.output_sigma_key][-idx]
                
                out_im_max = np.max(out_im)
                out_sigma_max = np.max(out_sigma)
                
                if out_im_max>2.25 or out_im_max<1.52:
                    sample_weight = high_temp_sample_weight
                else:
                    sample_weight = 1
                
                if Normalize_Output:
                    out_im = normalize_with_moments(out_im, means_vars['total_output'][0],means_vars['total_output'][1])
                # out_im = normalize_with_moments(out_im, means_vars[self.output_key][0],means_vars[self.output_key][1])
            
                if Inverse_Selection_of_Dataset:
                    if intensity<intensity_threshold or out_im_max>out_label_threshold or out_sigma_max>out_sigma_threshold: 
                        yield in_im, (out_im, out_sigma), sample_weight
                else:
                    if intensity_max>intensity>intensity_threshold and out_im_max<out_label_threshold and out_sigma_max<out_sigma_threshold: ### keeps 75% of images
                        yield in_im, (out_im, out_sigma), sample_weight

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






# %% 

""" Create Model"""

# # input = Input(shape =(input_arr.shape[1],input_arr.shape[2],1))
# input = Input(shape=image_shape)
# def norm_image(input):
#     normed = tf.image.per_image_standardization(input)
#     return normed
# x = Lambda(norm_image)(input)
# if Augment:
#     x = tf.keras.layers.RandomTranslation(height_factor=height_factor/(wavelength_end-wavelength_start),
#         width_factor=width_factor/(line_of_sight_end-line_of_sight_start),
#         fill_mode='nearest',
#         interpolation='nearest')(x)
#     x = tf.keras.layers.RandomRotation(
#         factor=rotation_factor,
#         fill_mode='nearest',
#         interpolation='nearest')(x)
# weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=123) #originally stddev was 0.01
# bias_initializer=tf.keras.initializers.Zeros()

# # Normalize input data to max of each individual image
# # max_val = tf.reduce_max(input)
# # x = tf.divide(input,max_val)
    
# x = Conv2D (filters =32, kernel_size =(3,3), strides = (3,3), padding ='same', kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x)
# x = BatchNormalization()(x)
# x = Activation(activ_fun)(x)
# x = MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same')(x)

# x = Conv2D (filters =64, kernel_size =kernel_size, strides = stride, padding ='same', kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x)
# x = BatchNormalization()(x)
# x = Activation(activ_fun)(x)
# x = MaxPool2D(pool_size =pool_size, strides =pool_stride, padding ='same')(x)

# # x = Flatten()(x)

# # #trying to incorporate attention layer
# # x = Dense(units=512)(x)
# if Transformer_On:
#     x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=24,attention_axes=(1, 2))(x,x)
# x = Flatten()(x)
# # x = tf.keras.layers.GlobalAveragePooling1D()(x)

# x = Dense(units = 128, activation =activ_fun, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 
# x = Dense(units = 80, activation =activ_fun, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 
# output = Dense(units = output_arr.shape[1], activation =None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 

# # creating the model

# model = Model (inputs=input, outputs =output)


#%%
# def run_training_plotting(model_num):
if True:
    if Sequential_Model:
        # MODEL_FNAME = base_dir+f"\\trained_model_{model_num}.h5"
        # # to be sure GPU memory is cleaned after last train
        # tf.keras.backend.clear_session()
        # print(f'################################################ Model_{model_num}')
        
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

                # model.add(Dropout(0.25))
                # model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
                # model.add(BatchNormalization(momentum=0.8))
                # model.add(LeakyReLU(alpha=0.2))

                #model.add(Dropout(0.25))
                model.add(Flatten())
                # model.add(Dense(units=512))
                # model.add(tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=512,value_dim=512,use_bias=False))
                # model.add(tf.keras.layers.GlobalAveragePooling1D())
                # model.add(Dense(256, activation='relu',kernel_regularizer=dense_regularizer))
                #model.add(BatchNormalization(momentum=0.8))
                # model.add(Dense(100, activation='relu',kernel_regularizer=dense_regularizer))
                # model.add(BatchNormalization(momentum=0.8))
                # model.add(Dropout(0.25))
                model.add(Dense(output_arr.shape[1], activation="linear",kernel_regularizer=dense_regularizer))

                return model
        
        model = build_model(image_shape)
    #%%
    def load_all_models():
        all_models = []
        model_names = [f'trained_model_{i}.h5' for i in np.arange(0,20)]
        # model_names = ['trained_model_huber.h5','trained_model_weighted_high_temp.h5', 'trained_model_32batch.h5']
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
        # ensemble_visible = [model.input for model in models]
        
        ensemble_outputs = [model(model_input) for model in models]
        print(model(model_input).shape)
        
        if Ensemble_Simple_Average:
            output = tf.keras.layers.Average()(ensemble_outputs)
        else:
            merge = tf.keras.layers.concatenate(ensemble_outputs, axis = 1)
            ###### double check correct reshape dimensions
            merge = tf.keras.layers.Reshape((output_arr.shape[1],len(models)))(merge) 
            #merge = tf.keras.layers.Reshape((len(models),output_arr.shape[1],1))(merge) 
            print(merge.shape)
            #merge = tf.keras.layers.Permute((2,1,3))(merge)
            local_2d = tf.keras.layers.LocallyConnected1D(1,kernel_size=len(models),padding='same',implementation=2,activation = 'linear')(merge)
            # local_2d = tf.keras.layers.LocallyConnected2D(1,kernel_size=(len(models),1),strides=(len(models),1),padding='same',implementation=2,activation = 'linear')(merge)
            print(local_2d.shape)
            output = tf.keras.layers.Flatten()(local_2d)
            output = tf.keras.layers.Dense(40,activation = 'linear')(output)
            print(output.shape)
        model = tf.keras.models.Model(inputs=model_input, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.Huber(huber_delta), metrics=[tf.keras.metrics.MeanSquaredError()],weighted_metrics=[])
        return model

    if Ensemble_Models:
        models = load_all_models()
        model = ensemble_model(models,Ensemble_Simple_Average)
        MODEL_FNAME = base_dir+f"\\trained_model_ensemble_{len(models)}.h5"

    if load_pretrained:
        
        model = tf.keras.models.load_model(saved_model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate,), #Adam originally
                            loss= loss_fn, #joint_loss,
                            metrics=[tf.keras.metrics.MeanSquaredError()],
                            weighted_metrics=[],
                            sample_weight_mode=[None])
    model.summary()

    """ Compile """


    #compile the model by determining loss function Binary Cross Entropy, optimizer as SGD
    if not Test_Only:
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
    """ Train """
    fully_trained = False

    if not Test_Only:
        history=model.fit(train_dataset,
            validation_data = valid_dataset,
            steps_per_epoch = steps,
            validation_steps = val_steps,
            epochs = num_epochs,
            verbose = 1,
            callbacks = [checkpointer,csv_logger,early_stopping,BestEpochCallback(), ClearMemory()]) #reduce_lr,schedule_callback, tensorboard_callback
        fully_trained = True

    # %% 
    """ Plot the train and validation Loss """
    if Plot_Metrics:

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
        # scores = model.evaluate(test_dataset,steps=int(1000/BATCH_SIZE))
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

        #plots per prediction
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
        
        plt.figure(figsize= (12,8))
        plt.imshow(tf.transpose(images[0,:,:,0]))
        plt.title(f'Example X-Ray Image')
        plt.xlabel('wavelength')
        plt.ylabel('line of sight')    

        # plt.figure()
        # plt.plot(rhos, np.mean(errors_per_rho_1000, axis=0))
        # plt.xlabel('rho')
        # plt.ylabel('MSE keV')
        # plt.title(f'Loss (MSE) at each rho [Raw Outputs]')
        # plt.savefig(save_plots_dir+r"\loss_per_rho.png")
        # plt.show(block=False)

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
            #thresholded_idx = np.where(intenstities<0.1e9)[0]
            #thresholded_idx = np.where(error>0.2)[0]
            i += 1 

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


        h = sns.jointplot(x=intenstities_all[:], y=errors_all[:],ratio=5,kind='hist') #,marginal_kws=dict(bins=20)
        # h.plot_joint(sns.kdeplot, color="r", zorder=0, levels=8)
        h.plot_marginals(sns.rugplot, color="r", height=-.1, clip_on=False)
        h.set_axis_labels('Intensity', 'MSE keV', fontsize=16)
        plt.suptitle(f'Loss (MSE) vs Intensity [Raw Outputs]')
        plt.tight_layout()
        plt.savefig(save_plots_dir+r"\loss_vs_intensity.png")
        plt.show(block=False)

        h = sns.jointplot(x=np.max(labels_all,axis=1), y=errors_all[:],ratio=5,kind='hist') #,marginal_kws=dict(bins=20)
        # h.plot_joint(sns.kdeplot, color="r", zorder=0, levels=8)
        h.plot_marginals(sns.rugplot, color="r", height=-.1, clip_on=False)
        h.set_axis_labels('Max Temp keV', 'MSE keV', fontsize=16)
        plt.suptitle(f'Loss (MSE) vs Max Temp [Raw Outputs]')
        plt.tight_layout()
        plt.savefig(save_plots_dir+r"\loss_vs_temp.png")
        plt.show(block=False)
            
        # find experiment number for corresponding bad profiles   
        num_bad=6    
        with h5py.File(hdf5_path, "r") as f:
            test_output_key = list(f.keys())[4]
            sorted_max_labels = np.sort(np.max(f[test_output_key][:],axis=1))
            print(sorted_max_labels.shape)
            sorted_max_idx = np.argsort(np.max(f[test_output_key][:],axis=1))
            max_labels = f[test_output_key][np.sort(sorted_max_idx[-num_bad:])[:-1]]
            mean_labels = np.mean(f[test_output_key][:], axis=0)
            #print(sorted_max_labels[-60:])
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
            #print(sorted_max_sigmas[-60:])

        plt.figure()
        # for i in range(len(max_labels)):
        #     plt.plot(rhos,max_labels[i]) 
        #     plt.fill_between(rhos, max_labels[i] - max_sigmas[i], max_labels[i] + max_sigmas[i],
        #                     color='gray', alpha=0.5)
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

            
        plt.figure() # for scatter plot
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

        #del model
        
# for model_num in np.arange(9,20): #stopped at 630 epochs for model_8
#     run_training_plotting(model_num)        
        
print("End of Training")
# %% 
import w7xarchive
import numpy as np
import matplotlib.pyplot as plt
# from skimage.measure import block_reduce
#%%
signal_name = "ArchiveDB/raw/W7X/ControlStation.71501/DETECTOR0-1_DATASTREAM/0/frames"
# time_from = '2022-11-09 00:00:00.00' #'YYYY-MM-DD HH:MM:SS.%f'
# time_to = '2022-11-10 00:00:00.00'
shotnum = '20221109.15'
time_intervals = w7xarchive.get_time_intervals_for_program(signal_name, shotnum)
# time_intervals = w7xarchive.get_time_intervals(signal_name, time_from, time_to)
# t, d = w7xarchive.get_image_json("ArchiveDB/raw/W7X/ControlStation.71501/DETECTOR0-1_DATASTREAM/0/frames", time_intervals[0], time_intervals[-1])
print(time_intervals.shape)

# time_intervals = w7xarchive.get_time_intervals_for_program(signal_name, "20221109.015") 

# # reads the timestamp (in nanoseconds) corresponding to the beginning of the discharge. we will download data from this point in time
# from_time = w7xarchive.get_program_t1("20221109.015")
# # we will read data until this time point. this is just the starting point + 1 second (converted to nanosecond, hence 1*10^9).
# # this is just an example; we can read data for much longer than 1 second, just replace as you need it
# to_time = from_time + 4 * 10**9

def moving_average(a, n=10) :
    ret = np.cumsum(a, axis =0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def moving_starts(a,n=10):
    return a[n-1:]

# def groupedAvg(myArray, N=10):
#     if myArray.shape[0]%10 != 0:
#         return block_reduce(myArray, block_size=(N,1,1), func=np.mean)[:-1]
#     else:
#         return block_reduce(myArray, block_size=(N,1,1), func=np.mean)
# def start_times(myArray, N=10):
#     if myArray.shape[0]%10 != 0:
#         return myArray[::N][:-1]
#     else:
#         return myArray[::N]

if Collect_New_Data:
    all_new_images =[]
    # raw_images = []
    times = []
    good_times = []

    for tim_e in time_intervals[:]: #index 757 error
        from_time = tim_e[0]
        to_time = tim_e[1]
        t, d = w7xarchive.get_image_json("ArchiveDB/raw/W7X/ControlStation.71501/DETECTOR0-1_DATASTREAM/0/frames", from_time, to_time)
        times.append(t)
        #raw_images.append(d)
        # print(t.shape)
        # print(d.shape)
        # plt.figure()
        # plt.imshow(d[0])
        if len(d.shape)>2:
            shot_avgs = moving_average(d)
            start_ts =moving_starts(t)
            # shot_avgs = groupedAvg(d)
            # start_ts = start_times(t)

            shot_avgs = shot_avgs[:,wavelength_start:wavelength_end,line_of_sight_start:line_of_sight_end]
                            
            new_images = shot_avgs.reshape(shot_avgs.shape[0],shot_avgs.shape[1],shot_avgs.shape[2],1)
            # plt.imshow(new_images[10,:,:,0])
            new_images[new_images>6e4]=1
            # plt.imshow(new_images[10,:,:,0])
            intensities = np.sum(new_images,axis=(1,2,3))
            good_new_images = np.where(intensities>intensity_threshold)
            if new_images[good_new_images].shape[0]>0:
                all_new_images.append(new_images[good_new_images])
                good_times.append(start_ts[good_new_images])
    new_good_times = np.concatenate(good_times)            
    new_times = np.concatenate(times)
    # new_raw_data = np.concatenate(raw_images)
    new_exp_data = np.concatenate(all_new_images)
    np.save(r'C:\Users\joaf\Documents\New_XICS_Image_times_091122_cropped.npy',new_good_times)
    np.save(r'C:\Users\joaf\Documents\New_XICS_Images_091122_cropped.npy',new_exp_data)
if not Collect_New_Data:
    new_exp_data = np.load(r'C:\Users\joaf\Documents\New_XICS_Images_091122.npy')
    new_good_times = np.load(r'C:\Users\joaf\Documents\New_XICS_Image_times_091122.npy')
#%% New Ti Profiles
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

# saved_model = 'C:\\Users\\joaf\\Documents\\models\\trained_model_cnn_fine_tuned.h5'
#saved_model = 'C:\\Users\\joaf\\Documents\\models\\trained_model.h5'

model = tf.keras.models.load_model(MODEL_FNAME)
num_plots = 4
fig, axs = plt.subplots(figsize= (10,16*ppp),nrows=num_plots, ncols = 2, gridspec_kw={'width_ratios': [1, 5]}, tight_layout =True)

i = 0 
# pred = model.predict(new_exp_data)
while True:
        rand = random.randrange(9,len(overlapping_times))
        if time[rand]-time[rand-10] == 1000000000:
            print(i)
            if i==num_plots:
                break
            # new_exp_idx = np.where(new_good_times == find_nearest(new_good_times, time[rand])[0] )
            # pred = model.predict(new_exp_data[new_exp_idx[0][0]:new_exp_idx[0][0]+1])
            axs[i][0].imshow(tf.transpose(new_exp_data[rand,:,:,0]))
            # axs[i*ppp+p+1].imshow(new_exp_data[new_exp_idx[0][0],:,:,0])
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
#tf.keras.backend.clear_session()
print('all done') 

# import visualkeras
# visualkeras.layered_view(model,legend=True, draw_volume=True)

# timestamps_img, values_img = w7xarchive.get_signal_for_program(signal_name,"20221109.15")
# timestamps_ti, values_ti = w7xarchive.get_signal_for_program(ti_signal_name,"20221109.15")
# %%
# %%
if Transformer_On:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    # Define the ImageTransformer model
    # transformer_model = Sequential([
    #     model,
    #     tf.keras.layers.Flatten(),
    #     Dense(units=512),
    #     tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=512),
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     Dense(units=40)
    # ])

    model.trainable = False
    Transformer_On = True
    base_model = tf.keras.Sequential()
    for layer in model.layers[:-2]:
        base_model.add(layer)
    base_model.trainable = False

    input = Input(shape=image_shape)
    def norm_image(input):
        normed = tf.image.per_image_standardization(input)
        return normed
    x = Lambda(norm_image)(input)
    if Augment:
        x = tf.keras.layers.RandomTranslation(height_factor=height_factor/(wavelength_end-wavelength_start),
            width_factor=width_factor/(line_of_sight_end-line_of_sight_start),
            fill_mode='nearest',
            interpolation='nearest')(x)
        x = tf.keras.layers.RandomRotation(
            factor=rotation_factor,
            fill_mode='nearest',
            interpolation='nearest')(x)
    weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=123) #originally stddev was 0.01
    bias_initializer=tf.keras.initializers.Zeros()

    # Normalize input data to max of each individual image
    # max_val = tf.reduce_max(input)
    # x = tf.divide(input,max_val)
        
    x = base_model(x)
    # x = Flatten()(x)

    # #trying to incorporate attention layer
    # x = Dense(units=512)(x)
    if Transformer_On:
        x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=40,attention_axes=(1, 2))(x,x)
        x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=24,attention_axes=(1, 2))(x,x)
    x = Flatten()(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = Dense(units = 128, activation =activ_fun, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 
    x = Dense(units = 80, activation =activ_fun, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 
    output = Dense(units = output_arr.shape[1], activation =None, kernel_regularizer=regularizer, kernel_initializer=weight_initializer,bias_initializer=bias_initializer)(x) 

    # creating the model

    transformer_model = Model (inputs=input, outputs =output)


    # Define the loss function and optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = Adam(learning_rate=1e-4)

    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate,), #Adam originally
                            loss= loss_fn, #joint_loss,
                            metrics=[tf.keras.metrics.MeanSquaredError()],
                            weighted_metrics=[],
                            sample_weight_mode=[None])
    transformer_model.summary()
    #%%
    history=transformer_model.fit(train_dataset,
        validation_data = valid_dataset,
        steps_per_epoch = steps,
        validation_steps = val_steps,
        epochs = 10,
        verbose = 1,
        callbacks = [checkpointer,csv_logger,early_stopping]) 
# # %%
# CONFIGURATION = {
#     "BATCH_SIZE": 32,
#     "IM_SIZE": 256,
#     "LEARNING_RATE": 1e-3,
#     "N_EPOCHS": 20,
#     "DROPOUT_RATE": 0.0,
#     "REGULARIZATION_RATE": 0.0,
#     "N_FILTERS": 6,
#     "KERNEL_SIZE": 3,
#     "N_STRIDES": 1,
#     "POOL_SIZE": 2,
#     "N_DENSE_1": 1024,
#     "N_DENSE_2": 128,
#     "NUM_CLASSES": 3,
#     "PATCH_SIZE": 16,
#     "PROJ_DIM": 768,
#     "CLASS_NAMES": ["angry", "happy", "sad"],
# }
# test_image = images[0]
# patches = tf.image.extract_patches(images=tf.expand_dims(test_image, axis = 0),
#                            sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
#                            strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
#                            rates=[1, 1, 1, 1],
#                            padding='VALID')

# print(patches.shape)
# patches = tf.reshape(patches, (patches.shape[0], -1, 768))
# print(patches.shape)

# plt.figure(figsize = (8,8))

# for i in range(patches.shape[1]):

#     ax = plt.subplot(16,16, i+1)
#     plt.imshow(tf.reshape(patches[0,i,:], (16,16,3)))
#     plt.axis("off")
    
# class PatchEncoder(tf.keras.layers.Layer):
#   def __init__(self, N_PATCHES, HIDDEN_SIZE):
#     super(PatchEncoder, self).__init__(name = 'patch_encoder')

#     self.linear_projection = Dense(HIDDEN_SIZE)
#     self.positional_embedding = tf.keras.layers.Embedding(N_PATCHES, HIDDEN_SIZE )
#     self.N_PATCHES = N_PATCHES

#   def call(self, x):
#     patches = tf.image.extract_patches(
#         images=x,
#         sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
#         strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
#         rates=[1, 1, 1, 1],
#         padding='VALID')
    
#     patches = tf.reshape(patches, (tf.shape(patches)[0], 256, patches.shape[-1]))
    
#     embedding_input = tf.range(start = 0, limit = self.N_PATCHES, delta = 1 )
#     output = self.linear_projection(patches) + self.positional_embedding(embedding_input)
    
#     return output

# class TransformerEncoder(Layer):
#   def __init__(self, N_HEADS, HIDDEN_SIZE):
#     super(TransformerEncoder, self).__init__(name = 'transformer_encoder')

#     self.layer_norm_1 = tf.keras.layers.LayerNormalization()
#     self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    
#     self.multi_head_att = tf.keras.layers.MultiHeadAttention(N_HEADS, HIDDEN_SIZE )
    
#     self.dense_1 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)
#     self.dense_2 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)
    
#   def call(self, input):
#     x_1 = self.layer_norm_1(input)
#     x_1 = self.multi_head_att(x_1, x_1)

#     x_1 = tf.keras.layers.Add()([x_1, input])

#     x_2 = self.layer_norm_2(x_1)
#     x_2 = self.dense_1(x_2)
#     output = self.dense_2(x_2)
#     output = tf.keras.layers.Add()([output, x_1])

#     return output

# class ViT(Model):
#   def __init__(self, N_HEADS, HIDDEN_SIZE, N_PATCHES, N_LAYERS, N_DENSE_UNITS):
#     super(ViT, self).__init__(name = 'vision_transformer')
#     self.N_LAYERS = N_LAYERS
#     self.patch_encoder = PatchEncoder(N_PATCHES, HIDDEN_SIZE)
#     self.trans_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
#     self.dense_1 = Dense(N_DENSE_UNITS, tf.nn.gelu)
#     self.dense_2 = Dense(N_DENSE_UNITS, tf.nn.gelu)
#     self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation = 'softmax')
#   def call(self, input, training = True):

#     x = self.patch_encoder(input)

#     for i in range(self.N_LAYERS):
#       x = self.trans_encoders[i](x)
#     x = Flatten()(x)
#     x = self.dense_1(x)
#     x = self.dense_2(x)
    
#     return self.dense_3(x)

# vit = ViT(
#     N_HEADS = 4, HIDDEN_SIZE = 768, N_PATCHES = 256,
#     N_LAYERS = 2, N_DENSE_UNITS = 128)
# vit(tf.zeros([2,256,256,3]))

# vit.summary()