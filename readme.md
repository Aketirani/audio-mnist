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
|   ├── *.yaml                  <-- Configuration Files
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
├── audio_mnist.py              <-- Main Python Script
|
├── audio_mnist.sh              <-- Main Shell Script
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
The choice of [XGBoost](https://xgboost.readthedocs.io/en/stable/) (eXtreme Gradient Boosting) is grounded in its efficiency and scalability, especially for large datasets and complex models. It stands out for its capacity to manage missing values, categorical variables, and high-dimensional data, making it aptly suited for the project. Moreover, it encompasses regularization techniques that deter overfitting and enhance model generalization. 

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

### Model Performance
The model has achieved a promising accuracy rate of `95.77%` on the test dataset, showcasing its potential in gender recognition through voice analysis.

### PostgreSQL Integration
PostgreSQL integration enhances the project's data management capabilities. By utilizing [PostgreSQL](https://www.postgresql.org), an open-source object-relational database system, we ensure scalability, robustness, and efficient storage and retrieval of data. Utilize psycopg2, a Python driver, for seamless interaction with PostgreSQL. Store connection details in the `config/postgres.yaml` file for easy access, and perform database operations using SQL queries.

### Conclusion
This project epitomizes the application of machine learning in voice and speech analysis, showcasing the potential to discern gender through acoustic features.

### Exectuion
Execute `audiomnist.sh` to initiate the entire pipeline.

Following arguments can be specified:

- `-c`, `--cfg_file`: Path to the configuration file
- `-y`, `--pgs_file`: Path to the PostgreSQL configuration file
- `-d`, `--data_prep`: Indicates whether to execute the data preparation step (true or false)
- `-f`, `--feat_eng`: Indicates whether to execute the feature engineering step (true or false)
- `-s`, `--data_split`: Indicates whether to execute the data splitting step (true or false)
- `-u`, `--model_tune`: Indicates whether to execute the model tuning step (true or false)
- `-t`, `--model_train`: Indicates whether to execute the model training step (true or false)
- `-p`, `--model_pred`: Indicates whether to execute the model prediction step (true or false)
- `-q`, `--data_sql`: Indicates whether to execute the data sql step (true or false)

### Unit Test
Execute `python -m unittest discover test` to run all unit tests, ensuring the reliability of the code base.

### Developer
Execute `python -m pre_commit run --all-files` to ensure code quality and formatting checks.
