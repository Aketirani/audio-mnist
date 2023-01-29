# Gender Recognition By Voice & Speech Analysis
A voice recording is analyzed for many inferences like the content spoken, the emotion, gender, and identity of the speaker, and many more.

While recognizing the characteristics of the speaker from the recording, identity recognition requires a reference dataset and is limited to identifying the people in the dataset. In many cases, this level of specificity may not be needed.

Recognizing the speaker’s gender is one such use case. The model for the same can be trained to learn patterns and features specific to each gender and reproduce it for individuals who are not part of the training dataset too.

It finds applications in automatic salutations, tagging audio recording, and helping digital assistants reproduce male or female generic results.

In this repository, we will be using acoustic features extracted from a voice recording to predict the speaker’s gender.

## Structure
```
├── readme.md                   <-- You Are Here
|
├── data                        <-- Data Folder
|   ├── audioMNIST_meta.txt     <-- Meta Information
|   ├── number                  <-- Folder Number
|       ├── *.wav               <-- Audio Recordings
|
├── data_pre                    <-- Preprocced Data Folder
|   ├── features_data.csv       <-- Features Dataset CSV File
|   ├── final_data.csv          <-- Final Dataset CSV File
|
├── model_param
|   ├── *.yaml                  <-- Model Parameters
|
├── plots
|   ├── *.png                   <-- Plots
|
├── results
|   ├── *.yaml                  <-- Model Results
|
├── source
|   ├── config.yaml             <-- Configuration File Including Paths
|   ├── data_processing.py      <-- Data Processing
|   ├── data_split.py           <-- Data Split
|   ├── data_visualization.py   <-- Data Visualization
|   ├── feature_engineering.py  <-- Feature Engineering
|   ├── main.ipynb              <-- Executing Whole Pipeline
|   ├── model_parameters.yaml   <-- Model Parameters
|   ├── setup.py                <-- Setup
|   ├── utilites.py             <-- Utilities
|   ├── xgboost_model.py        <-- Model Training And Prediction
```

## Dataset
[Kaggle](https://www.kaggle.com/datasets/primaryobjects/voicegender)

The dataset consists of 30.000 audio samples of spoken digits (0-9) of 60 different speakers.

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

- learning_rate (eta): The step size of the optimization algorithm
- max_depth: The maximum depth of a tree
- n_estimators: The number of of trees in the ensemble
- num_parallel_tree: Number of parallel trees to be built
- gamma: Minimum split loss
- lambda: Regularization term
- scale_pos_weight: The balance between positive and negative weights
- min_child_weight: Minimum sum of weights of all observations in a child
- objective: Loss function
- tree_method: Method used to grow the tree
- verbosity: Level of verbosity of printing messages

By adjusting these parameters, the model can be fine-tuned to achieve the best performance on the given dataset.

## Model Features
The features are calculates by various statistical features of the given audio data and FFT data.

These features include:

- Mean of the data
- Standard deviation of the data
- Median of the data
- 25th percentiles of the data
- 75th percentiles of the data
- Minimum value of the data
- Maximum value of the data
- Skewness of the data
- Kurtosis of the data
- Range of the data

These features are stored in a dictionary, where each key corresponds to a specific feature and its associated value. This dictionary can then be used as input for the model.

## Exectuion
Run `main.ipynb` script to execute whole pipeline
