# Gender Recognition By Voice Analysis
This repository is dedicated to a project that leverages acoustic features derived from voice recordings to predict the gender of the speaker. The analysis is based on a collection of 30,000 audio samples and utilizes the XGBoost machine learning library to achieve this objective. 

### Project Overview
Voice recordings can be analyzed to infer a myriad of details including the content spoken, the emotional state, gender, and even the identity of the speaker. This project centers on gender recognition, aiming to create a model capable of identifying gender-specific patterns and features in voice recordings.

### Structure
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
|   ├── *.csv                   <-- Data Files
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
|   ├── *.py                    <-- Source Files
|
├── test                        <-- Test Folder
|   ├── *.py                    <-- Unit Tests
|
├── .gitignore                  <-- Configuring Ignored Files
|
├── .pre-commit-config.yaml     <-- Pre-Commit Configuration
|
├── audiomnist.py               <-- Main Python Script
|
├── audiomnist.sh               <-- Main Shell Script
|
├── flowchart.wsd               <-- Pipeline Flowchart
|
├── readme.md                   <-- You Are Here
|
├── requirements.txt            <-- Package Requirements
```

### Dataset
This project utilizes a comprehensive [dataset](https://www.kaggle.com/datasets/primaryobjects/voicegender) encompassing 30,000 spoken digits (0-9) audio samples from 60 distinct speakers. The dataset includes a directory for each speaker containing their respective audio recordings and a meta-information file detailing the gender and age of each speaker.

### Model Selection
The choice of XGBoost (eXtreme Gradient Boosting) is grounded in its efficiency and scalability, especially for large datasets and complex models. It stands out for its capacity to manage missing values, categorical variables, and high-dimensional data, making it aptly suited for the project. Moreover, it encompasses regularization techniques that deter overfitting and enhance model generalization. 

### Model Architecture
The project employs a gradient boosting tree ensemble model within the XGBoost framework. This model synergizes predictions from several decision trees to arrive at a final prediction for the target variable — the speaker's gender. A set of key parameters, including learning rate, maximum tree depth, and number of trees in the ensemble, govern the model, allowing for fine-tuning to attain optimal performance on the dataset.

The model uses the following key parameters:

- `learning_rate`: The step size of the optimization algorithm
- `max_depth`: The maximum depth of a tree
- `n_estimators`: The number of of trees in the ensemble
- `objective`: Loss function
- `tree_method`: Method used to grow the tree

### Model Features
The model relies on various statistical features calculated from both the time-domain and frequency-domain data of the audio samples. These features are crucial in training the model to recognize gender-specific patterns in the voice recordings.

These features include:

- `mean`: Mean
- `std`: Standard Deviation
- `med`: Median
- `q25`: 25th Percentiles
- `q75`: 75th Percentiles
- `min`: Minimum
- `max`: Maximum
- `skew`: Skewness
- `kurt`: Kurtosis
- `zeroxrate`: Zero Crossing Rate
- `entropy`: Entropy
- `sfm`: Spectral Flatness
- `cent`: Frequency Centroid

These features are stored in a dictionary, where each key corresponds to a specific feature and its associated value. This dictionary can then be used as input for the model.

### Model Performance
With meticulous tuning and optimization, the model has achieved a promising accuracy rate of `87.40%` on the test dataset, showcasing its potential in gender recognition through voice analysis.

### Exectuion
Execute `audiomnist.sh` to initiate the entire pipeline.

### Unit Test
Execute `python -m unittest discover test` to run all unit tests, ensuring the reliability of the code base.

### Developer
Execute `python -m pre_commit run --all-files` to ensure code quality and formatting checks.

### Conclusion
This project epitomizes the application of machine learning in voice and speech analysis, showcasing the potential to discern gender through acoustic features.
