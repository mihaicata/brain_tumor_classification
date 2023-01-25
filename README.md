# brain_tumor_classification
## Applied Deep Learning Assignment 1
## Mihai Cata 12045439

### Topic of choice
For this project I will choose the topic Computer Vision - Image classification.
### Project type
The project type I want to choose is "Bring your own method"
### Project description
My project consists of usind Deep Learning to classify images of brain tumors. My ideas so far (from the literature review I've done) is to try and use methods like Convolutional Neural Networks and Transfer Learning for this approach, and try to get an accuracy as high as possible for the dataset I will be working on.
### Dataset description
I have searched for multiple datasets, and they usually consist of images having 4 possible classes, corresponding to the kind of tumor present, or no tumor if the patient is healthy. The dataset I like the most and will use for my experiments contains 3275 such images, having different resolutions, which is already a challenge for the preprocessing step.
### Work-breakdown structure
My indended planning looks like this:
* dataset collection and preprocessing - 2 days
* designing and building an appropriate network - 3 days
* training and fine-tuning that network - 8 days
* building an application to present the results -  3 days
* writing the final report - 1 day
* preparing the presentation of your work 1 day
### Literature
1. Alqudah, Ali Mohammad et al. “Brain Tumor Classification Using Deep Learning Technique - A Comparison between Cropped, Uncropped, and Segmented Lesion Images with Different Sizes.” ArXiv abs/2001.08844 (2019)
2. Díaz-Pernas FJ, Martínez-Zarzuela M, Antón-Rodríguez M, González-Ortega D. A Deep Learning Approach for Brain Tumor Classification and Segmentation Using a Multiscale Convolutional Neural Network. Healthcare (Basel). 2021 Feb 2;9(2):153.
3. Ullah, Naeem & Khan, Javed & Khan, Mohammad & Khan, Wahab & Hassan, Izaz & Obayya, Marwa & Negm, Noha & Salama, Ahmed. (2022). An Effective Approach to Detect and Identify Brain Tumors Using Transfer Learning. Applied Sciences.

## Applied Deep Learning Assignment 2

Chosen error metric: Accuracy
Error metric target to achieve: accuracy>=70%
Actually achieved accuracy: 92%
Time spent for each task:
* data collection: 2 hours
* data preprocessing: 8 hours
* model experiments: 16 hours
* hyper parameter tuning: 32 hours
* data augmentation + experiments: 10 hours
* documentation + submission: 3 hours

### Summary

I used Jupyter Notebooks for my experiments, since I believed it will make the code easier to go through and comment and easier to reproduce. In order to facilitate reproducing the code, I created a pipeline of Jupyter scripts, that, if run in the correct order, show the entire Data Science pipeline I went through. The files are structured as follow:
* Script 1: Used for data exploration (gaining some insights about it)
* Script 2: Used for data preprocessing (preparing it for training)
* Script 3: Used for experiments with different models, approaches, hyper parameter tuning
* Script 4: Used for exploration and preprocessing on the augmented data
* Script 5: Used for experiments on the augmented dataset

### About the experiments

For the experiments, I tried to use all of the approaches you indicated (ex hyperparameter optimization, data augmentation, different model architectures, transfer learning etc.). There has been more experiments performed than the ones present on the final jupyter script (a lot of failed ones or not very relevant ones). In an attempt to make things clearer, I will sum up in a table some of the best experiments and their results.

| Network Architecture        | Number of epochs | Accuracy |
|-----------------------------|------------------|----------|
| CNN_2_layers                | 50               | 0.7360   |
| CNN_3_layers                | 100              | 0.7664   |
| Resnet_transfer_learning    | 50               | 0.7563   |
| Resnet_transfer_learning    | 20               | 0.7487   |
| Resnet_trainable_network    | 50               | 0.7360   |
| Resnet_trainable_network    | 20               | 0.7461   |
| CNN_3_layers_augmented_data | 50               | 0.9202   |

Among the hyperparameters I tried to tune, there are:
* learning rate
* optimizer
* number of epochs
* number of layers in the model

There were a lot of computation power impediments that didn't allow me to perform more complex experiments, but I tried to overcome this and make the best out of my conditions.

### How to run the experiments
1. Download the data. First dataset used from [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?resource=download). Second dataset used consists of a combination of the first dataset and [this dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download) (I combined them manually, takes less than a minute).
2. Create an Python environment and use the *requirements.txt* file to install all the required packages.
3. Run the scripts in order (1->5). The only thing that should be changed are the file locations, which are held in variables, should be very visible and intuitive.


## Applied Deep Learning Assignment 3

### How to run the demo web app

1. Install flask in your python environment
2. Open a terminal inside the folder *demo_web_app*
3. Run *python app.py* in the terminal
4. Open the browser address specified in the terminal and play around with the app
