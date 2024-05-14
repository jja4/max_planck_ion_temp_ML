## README: XICS Image Classification for Ion Temperature Prediction

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

