## README 
## A machine learning pipeline for predicting ion temperature profiles in W7-X stellarator discharges using X-ray images.

---
## Ion Temperature Data Processing for X-Ray Image Classification (import_raw_data.py)

This code prepares an HDF5 dataset containing X-ray images and corresponding ion temperature profiles for training a machine learning model. 

**Data Source:**

* X-ray image archives located at `\\x-drive\Diagnostic-logbooks\QSW-xRayImaging\w7x_ar16`.
* Ion temperature data retrieved from the MDSplus server using `qsw_eval` program number.

**Process:**

1. **Gather Data Names:**
    * Identifies all eligible data shots from zip files within the specified directory.
    * Extracts shot numbers and file paths for X-ray images.
2. **Connect to MDSplus:**
    * Sets the environment variable for the MDSplus server connection.
3. **Process Shots:**
    * Iterates through identified shots.
    * Attempts to connect to the MDSplus tree for each shot and retrieve ion temperature data, masks, sigmas, and time information.
    * Filters shots based on data quality (percentage of valid data points in the mask).
    * Classifies shots as successfully read, misread, not accessible (read-only), or containing errors.
4. **Prepare Dataset:**
    * Groups usable X-ray image sets with corresponding ion temperature profiles and sigmas based on shot number.
    * Defines a function `groupedAvg` to perform block-wise averaging on the image data (configurable block size).
    * Creates an HDF5 file and stores training, validation, and test sets:
        * X-ray images (after averaging) are stored in `train_input_image`, `valid_input_image`, and `test_input_image` datasets.
        * Ion temperature profiles are stored in `train_target_image`, `valid_target_image`, and `test_target_image` datasets.
        * Ion temperature sigmas are stored in `train_target_sigma`, `valid_target_sigma`, and `test_target_sigma` datasets.
    * Randomly assigns data points to training, validation, and test sets with a probability distribution of 80% training, 10% validation, and 10% test.
    * Performs basic data shape validation.
5. **Report and Save:**
    * Prints informative messages about the number of processed shots, successful data retrievals, data filtering results, and the number of data points in each set of the HDF5 file.
    * Saves the HDF5 dataset to the specified directory (`C:\Users\joaf\Documents\Ion_Temp_Dataset_adj.h5`).

**Outputs:**

* An HDF5 file containing pre-processed X-ray images, ion temperature profiles, and sigmas split into training, validation, and test sets.

**Notes:**

* The code handles potential errors during data access and filtering.
* It identifies shots with mismatched time resolution between X-ray images and ion temperature data (not 100 Hz).
* Modify the script variables like `rootdir` and `savedir` to point to your data and desired output locations.

---

## XICS Image Classification for Ion Temperature Prediction (ion_temp_cnn.py)

This code trains a Convolutional Neural Network (CNN) to predict ion temperature profiles from X-ray images collected during plasma discharges in the W7-X stellarator.
The ground truth ion temperature profiles are generated with the Levenberg-Marquadt Minimization method by Novimir Pablant. 
The code also includes functionalities to collect new data and visualize predictions on unseen data.

**Requirements:**

* TensorFlow ([https://www.tensorflow.org/](https://www.tensorflow.org/))
* h5py ([https://docs.h5py.org/en/stable/high/group.html](https://docs.h5py.org/en/stable/high/group.html))
* matplotlib ([https://matplotlib.org/](https://matplotlib.org/))
* seaborn (optional, for plotting)
* Other libraries: random, numpy, gc

**Data Path:**

* The code creates a dataset in the HDF5 format in the import_raw_data.py file.
* In the ion_temp_cnn.py file, update the `hdf5_path` variable to point to your data file.

**Running the code:**

1. Modify the hyperparameters in the script according to your needs. 
2. Set `Collect_New_Data` to `True` if you want to collect new data from the W7-X archive. Update `time_intervals` and other data access variables accordingly. 
3. Run the script.

**Explanation of the Code:**

* **Data Preparation:**
    * Loads data from the HDF5 file based on user-defined ROI and filtering criteria.
    * Creates data generators for efficient training.
* **Model Building:**
    * Builds a sequential CNN model with convolutional layers, activation functions, pooling layers, dropout (optional), and a final dense layer.
    * Alternatively, builds an ensemble model by combining multiple pre-trained models.
* **Training (if not in test mode):**
    * Defines the optimizer, loss function, metrics, and early stopping criteria.
    * Trains the model on the prepared data.
* **New Data Collection (if Collect_New_Data is True):**
    * Connects to the W7-X archive and retrieves X-ray images for the specified time intervals.
    * Applies filtering based on intensity and other criteria.
    * Saves the collected data as `.npy` files for future use.
* **Prediction on New Data:**
    * Loads a pre-trained model.
    * Selects random samples from the collected new data with overlapping time stamps in the available temperature data.
    * Generates plots comparing the predicted temperature profiles from the model with the actual temperature profiles retrieved from the archive.
    * Saves the plots.

