# Gender Recognition By Voice & Speech Analysis
A voice recording is analyzed for many inferences like the content spoken, the emotion, gender, and identity of the speaker, and many more.

While recognizing the characteristics of the speaker from the recording, identity recognition requires a reference dataset and is limited to identifying the people in the dataset. In many cases, this level of specificity may not be needed.

Recognizing the speaker’s gender is one such use case. The model for the same can be trained to learn patterns and features specific to each gender and reproduce it for individuals who are not part of the training dataset too.

In this repository, we will be using acoustic features extracted from a voice recording to predict the speaker’s gender.

## Structure
```
├── audio                       <-- Audio Folder
|   ├── recordings              <-- Recordings Folder
|       ├── *.wav               <-- Audio Recordings
|   ├── audioMNIST_meta.txt     <-- Meta Information
|
├── config                      <-- Configuration Folder
|   ├── config.yaml             <-- Configuration File
|
├── data                        <-- Data Folder
|   ├── features_data.csv       <-- Features Dataset
|   ├── final_data.csv          <-- Final Dataset
|
├── logs                        <-- Log Folder
|   ├── *.log                   <-- Log Files
|
├── parameters                  <-- Parameters Folder
|   ├── *.yaml                  <-- Model Parameters
|
├── plots                       <-- Plots Folder
|   ├── *.png                   <-- Plots
|
├── results                     <-- Results Folder
|   ├── *.yaml                  <-- Model Results
|
├── src                         <-- Source Folder
|   ├── data_processing.py      <-- Data Processing
|   ├── data_split.py           <-- Data Split
|   ├── data_visualization.py   <-- Data Visualization
|   ├── feature_engineering.py  <-- Feature Engineering
|   ├── setup.py                <-- Set Up Paths
|   ├── utilites.py             <-- Help Functions
|   ├── xgboost_model.py        <-- Model Training And Prediction
|
├── test                        <-- Test Folder
|   ├── *.py                    <-- Unit Tests
|
├── .gitignore                  <-- Configuring Ignored Files
|
├── .pre-commit-config.yaml     <-- Pre-Commit Configuration
|
├── readme.md                   <-- You Are Here
|
├── flowchart.wsd               <-- Pipeline Flowchart
|
├── main.py                     <-- Main Python Script
|
├── main.sh                     <-- Main Shell Script
|
├── requirements.txt            <-- Package Requirements
```

## Dataset
The [dataset](https://www.kaggle.com/datasets/primaryobjects/voicegender) consists of 30.000 audio samples of spoken digits (0-9) of 60 different speakers.

There is one directory per speaker holding the audio recordings.

Additionally "audioMNIST_meta.txt" provides meta information such as gender or age of each speaker.

## Model Selection
XGBoost (eXtreme Gradient Boosting) is a powerful and popular machine learning library for gradient boosting. It is designed to be efficient and scalable, and is particularly useful for large datasets and complex models.

One of the key advantages of XGBoost is its ability to handle missing values and categorical variables, which makes it a great choice for datasets like ours that may have missing or categorical data. Additionally, XGBoost is known for its ability to handle high dimensional data, which is useful for our audio classification task as we are using a large number of features extracted from the audio recordings.

XGBoost also includes a number of regularization techniques, such as L1 and L2 regularization, which can help prevent overfitting and improve the generalizability of the model.

Another important advantage of XGBoost is its ability to handle non-linear relationships between the features and the target variable, which is useful for our task as the relationship between the audio features and the speaker's gender may not be linear.

In summary, XGBoost is a powerful and versatile tool that is well-suited for our audio classification task due to its ability to handle high dimensional data, missing values, categorical variables, and non-linear relationships.

## Model Architecture
The XGBoost model used in this project is a gradient boosting tree ensemble model. The architecture of this model is composed of several decision trees, each of which is trained to make a prediction for the target variable (speaker's gender). The final prediction is made by combining the predictions of all the individual trees.

The model uses the following key parameters:

- `learning_rate`: The step size of the optimization algorithm
- `max_depth`: The maximum depth of a tree
- `n_estimators`: The number of of trees in the ensemble
- `gamma`: Minimum split loss
- `lambda`: Regularization term
- `scale_pos_weight`: The balance between positive and negative weights
- `min_child_weight`: Minimum sum of weights of all observations in a child
- `objective`: Loss function
- `tree_method`: Method used to grow the tree

By adjusting these parameters, the model can be fine-tuned to achieve the best performance on the given dataset.

## Model Features
The features are calculates by various statistical features of the given FFT data.

These features include:

- `mean`: Mean of the FFT data
- `std`: Standard deviation of the FFT data
- `med`: Median of the FFT data
- `q25`: 25th percentiles of the FFT data
- `q75`: 75th percentiles of the FFT data
- `min`: Minimum value of the FFT data
- `max`: Maximum value of the FFT data
- `skew`: Skewness of the FFT data
- `kurt`: Kurtosis of the FFT data
- `sfm`: Spectral flatness of the FFT data
- `cent`: Frequency centroid of the FFT data

These features are stored in a dictionary, where each key corresponds to a specific feature and its associated value. This dictionary can then be used as input for the model.

## Model Performance
Accuracy on test dataset: `83.27%`.

## Exectuion
Run `main.sh` to execute whole pipeline.

## Unit Test
Run `python -m unittest` to run all unit tests.
