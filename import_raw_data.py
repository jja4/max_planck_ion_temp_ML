# %% 
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import io
from PIL import Image
import MDSplus
import h5py
#%matplotlib inline


# %% 
print('gathering data names')
#get list of all eligible data shots
rootdir = r'\\x-drive\Diagnostic-logbooks\QSW-xRayImaging\w7x_ar16'
shots_17 = [x[9:-4] for x in map(os.path.basename, glob.glob(os.path.join(rootdir,'17[1-9]*','*.zip')))]
shots_18 = [x[9:-4] for x in map(os.path.basename, glob.glob(os.path.join(rootdir,'18*','*.zip')))]
shots = shots_17 + shots_18

#gather x-ray image paths for input to the network
xray_17 = [x for x in glob.glob(os.path.join(rootdir,'17[1-9]*','*.zip'))]
xray_18 = [x for x in glob.glob(os.path.join(rootdir,'18*','*.zip'))]
xray_zips = xray_17 + xray_18
# %% 

print('getting started')
# Set the path to the MDSPlus server.
MDSplus.setenv('qsw_eval_path', 'mds-data-1.ipp-hgw.mpg.de::')

#gather ion temp profiles for output/true values of the network
#ion_temp = np.empty((1,40))
ion_temp = []
ion_temp_time = []
sigmas = []
openable_shots = []
good_shots = []
read_only_shots = []
misreads = []
for shot in shots[:]:
    try:
        # Open the tree for evaluated XICS data for program 20180919.040
        # Here we used the 9 digit MDSplus program number.
        tree = MDSplus.Tree('qsw_eval', int(shot))
        openable_shots.append(shot)
        try:
            # Get the ion temperature.
            ti = tree.getNode('XICS:TI').data()
            ti_line = tree.getNode('XICS_LINE:TI').data()

            ti_mask = tree.getNode('XICS.TI:MASK').data().astype(bool)
            ti_sigma = tree.getNode('XICS.TI:SIGMA').data()
            ti_sigma_line = tree.getNode('XICS_LINE.TI:SIGMA').data()
            time_line = tree.getNode('XICS_LINE.TI:TIME').data()
            time = tree.getNode('XICS.TI:TIME').data()
            reff = tree.getNode('XICS.TI.REFF').data()
            rho = tree.getNode('XICS.TI.RHO').data()
            
            perc_good = sum(ti_mask)/ti_mask.shape[0]
            ion_temp.append(ti[:,perc_good>0.80].T)
            ion_temp_time.append(ti_line.shape[1])
            sigmas.append(ti_sigma[:,perc_good>0.80].T)
            good_shots.append(shot)
            #ion_temp = np.concatenate((ion_temp,ti[:,perc_good>0.80].T))
            print('YAY '+shot)
        except:
            print('misread '+shot)
            misreads.append(shot)            
    except:
        print('NEE '+shot)
        read_only_shots.append(shot)
target_images = np.array(ion_temp)
target_sigmas = np.array(sigmas)
numtargets_pershot = [i.shape[0] for i in target_images]
    
print('We have '+str(len(target_images))+' ion profiles')

num_nonzero_targets = [pershot for pershot in numtargets_pershot if pershot != 0]
idx_nonzero_targets = [idx for idx, pershot in enumerate(numtargets_pershot) if pershot != 0]

good_profile_shots = [good_shots[i] for i in idx_nonzero_targets]
good_shot_time_nums = [ion_temp_time[i] for i in idx_nonzero_targets]
good_target_images = target_images[idx_nonzero_targets]
good_sigmas = target_sigmas[idx_nonzero_targets]

print('We have '+str(len(num_nonzero_targets))+' non-zero ion profiles')
# %% 
#find usable xray images
print('beginning to assemble images and ion temp profiles into dataset file')

xrays_good_profile = [x for x in xray_zips if os.path.basename(x)[9:-4] in good_profile_shots]

print('Verify all these numbers are the same. Correct errror if not.')
print(len(num_nonzero_targets),len(good_profile_shots),len(xrays_good_profile),len(good_target_images),len(good_sigmas))


def groupedAvg(myArray, N=10):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result

import random

complete_shots =[]
incomplete_shots =[]
matching100Hz_shots = []
non100Hz_shots = []
numimages_pershot = []

""" Store an array of images to HDF5.
    Parameters:
    ---------------
    images       training_image array, (N, 195,1475) to be stored
    labels       target_image array, (N, 40) to be stored
"""

# Create a new HDF5 file
savedir = r'C:\Users\joaf\Documents'
file = h5py.File(os.path.join(savedir,'Ion_Temp_Dataset_adj.h5'), "w")


for count, zip in enumerate(xrays_good_profile[:]):
    archive = zipfile.ZipFile(zip, 'r')
    tifs = [tif for tif in archive.namelist() if '.tif' in tif]
    numimages_thisshot = len(tifs)
    numimages_pershot.append(numimages_thisshot)
    if numimages_thisshot % 100 == 0:
        complete_shots.append(zip[-13:-4])
        shot_images = []
        if good_shot_time_nums[count]*10 == numimages_thisshot:
            matching100Hz_shots.append(zip[-13:-4])
            # print('GOOD ' + zip[-13:-4])
            
            for timestep in tifs[:int(num_nonzero_targets[count]*10)]:
                imgdata = archive.read(timestep)
                dataEnc = io.BytesIO(imgdata)
                im = Image.open(dataEnc)
                imarray = np.array(im)
                shot_images.append(imarray)
            shot_avgs = groupedAvg(np.array(shot_images))
            
            
            training_image = np.array(shot_avgs)
            target_image = good_target_images[count]
            target_sigma = good_sigmas[count]
            if count == 0:
                # Create the dataset at first
                file.create_dataset('train_input_image', data=training_image, compression="gzip", chunks=True, maxshape=(None,training_image.shape[1],training_image.shape[2]))
                file.create_dataset('train_target_image', data=target_image, compression="gzip", chunks=True, maxshape=(None,target_image.shape[1]))
                file.create_dataset('train_target_sigma', data=target_sigma, compression="gzip", chunks=True, maxshape=(None,target_sigma.shape[1]))
            elif count == 1:
                file.create_dataset('valid_input_image', data=training_image, compression="gzip", chunks=True, maxshape=(None,training_image.shape[1],training_image.shape[2]))
                file.create_dataset('valid_target_image', data=target_image, compression="gzip", chunks=True, maxshape=(None,target_image.shape[1])) 
                file.create_dataset('valid_target_sigma', data=target_sigma, compression="gzip", chunks=True, maxshape=(None,target_sigma.shape[1]))
            elif count == 2:
                file.create_dataset('test_input_image', data=training_image, compression="gzip", chunks=True, maxshape=(None,training_image.shape[1],training_image.shape[2]))
                file.create_dataset('test_target_image', data=target_image, compression="gzip", chunks=True, maxshape=(None,target_image.shape[1])) 
                file.create_dataset('test_target_sigma', data=target_sigma, compression="gzip", chunks=True, maxshape=(None,target_sigma.shape[1]))
            else:
                # Append new data to it
                rand_num = random.random() 
                if (rand_num < 0.8):
                    file['train_input_image'].resize((file['train_input_image'].shape[0] + training_image.shape[0]), axis=0)
                    file['train_input_image'][-training_image.shape[0]:] = training_image

                    file['train_target_image'].resize((file['train_target_image'].shape[0] + target_image.shape[0]), axis=0)
                    file['train_target_image'][-target_image.shape[0]:] = target_image
                    
                    file['train_target_sigma'].resize((file['train_target_sigma'].shape[0] + target_sigma.shape[0]), axis=0)
                    file['train_target_sigma'][-target_sigma.shape[0]:] = target_sigma
                elif (0.9 > rand_num > 0.8):
                    file['valid_input_image'].resize((file['valid_input_image'].shape[0] + training_image.shape[0]), axis=0)
                    file['valid_input_image'][-training_image.shape[0]:] = training_image

                    file['valid_target_image'].resize((file['valid_target_image'].shape[0] + target_image.shape[0]), axis=0)
                    file['valid_target_image'][-target_image.shape[0]:] = target_image
                    
                    file['valid_target_sigma'].resize((file['valid_target_sigma'].shape[0] + target_sigma.shape[0]), axis=0)
                    file['valid_target_sigma'][-target_sigma.shape[0]:] = target_sigma
                else:
                    file['test_input_image'].resize((file['test_input_image'].shape[0] + training_image.shape[0]), axis=0)
                    file['test_input_image'][-training_image.shape[0]:] = training_image

                    file['test_target_image'].resize((file['test_target_image'].shape[0] + target_image.shape[0]), axis=0)
                    file['test_target_image'][-target_image.shape[0]:] = target_image
                    
                    file['test_target_sigma'].resize((file['test_target_sigma'].shape[0] + target_sigma.shape[0]), axis=0)
                    file['test_target_sigma'][-target_sigma.shape[0]:] = target_sigma
            if target_image.shape[0] != training_image.shape[0]:
                print("ERROR: input_image and target_image dimensions do not match!")
            print("I am on shot {} and 'train_input_image' chunk has shape: {} and 'train_target_image' chunk has shape: {}".format(zip[-13:-4],file['train_input_image'].shape,file['train_target_sigma'].shape))
            
        else:
            non100Hz_shots.append(zip[-13:-4])
            print('NOT 100Hz ' + zip[-13:-4])
    else:
        incomplete_shots.append(zip[-13:-4])
        print('dropped ' + zip[-13:-4])

file.close()
print('We have '+str(len(matching100Hz_shots))+' 100Hz shots with matching ion profiles and xray images')


print('all done')

# %% Extras
# #plot ion temperatures with mask
# for i in [0,25,50,75]:
#     st = i
#     en = i+25
#     plt.figure()
#     plt.plot(ti_mask[:,st:en]*ti[:,st:en])
#     plt.title('Seconds '+ str(st)+ ' to '+ str(en))
#     plt.show()


