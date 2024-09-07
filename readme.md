# Gender Recognition By Voice Analysis
This project focuses on leveraging acoustic features extracted from voice recordings to predict the gender of the speaker. Utilizing a large dataset and the XGBoost machine learning library, the project aims to develop a robust model for gender classification.

### Table of Contents
- [Project Overview](#project-overview)
- [Structure](#structure)
- [Dataset](#dataset)
- [Model Selection](#model-selection)
- [Model Architecture](#model-architecture)
- [Model Features](#model-features)
- [Model Performance](#model-performance)
- [PostgreSQL Integration](#postgresql-integration)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Execution](#execution)
- [Unit Tests](#unit-tests)
- [Developer](#developer)

### Project Overview
Voice recordings can be analyzed to infer a wide range of details including the content spoken, the emotional state, gender, and even the identity of the speaker. This project centers on gender recognition, aiming to create a model capable of identifying gender-specific patterns and features in voice recordings.

### Structure
```
├── audio                       <-- Audio Folder
|   └── recordings              <-- Recordings Folder
|       └── *.wav               <-- Audio Recordings
|   └── audioMNIST_meta.txt     <-- Meta Information
|
├── config                      <-- Configuration Folder
|   └── *.yaml                  <-- Configuration Files
|
├── data                        <-- Data Folder
|   └── *.csv                   <-- Data Files
|
├── images                      <-- Image Folder
|   └── *.png                   <-- Image Files
|
├── logs                        <-- Log Folder
|   └── *.log                   <-- Log Files
|
├── parameters                  <-- Parameters Folder
|   └── *.yaml                  <-- Model Parameters
|
├── plots                       <-- Plots Folder
|   └── *.png                   <-- Plots
|
├── results                     <-- Results Folder
|   └── *.yaml                  <-- Model Results
|
├── src                         <-- Source Folder
|   └── *.py                    <-- Source Files
|
├── test                        <-- Test Folder
|   └── *.py                    <-- Unit Tests
|
├── text                        <-- Text Folder
|   └── *.txt                   <-- Text Files
|
├── .gitignore                  <-- Git Ignore Configuration
|
├── .pre-commit-config.yaml     <-- Pre-Commit Configuration
|
├── audio_mnist.py              <-- Main Python Script
|
├── audio_mnist.sh              <-- Main Shell Script
|
├── flowchart.wsd               <-- Pipeline Flowchart
|
├── gui.py                      <-- GUI Python Script
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
The project employs a gradient boosting tree ensemble model within the XGBoost framework. This model synergizes predictions from several decision trees to arrive at a final prediction for the target variable — the speaker's gender. A set of key parameters govern the model, allowing for fine-tuning to attain optimal performance on the dataset.

The model uses the following key parameters:
| Parameter       | Description          |
|-----------------|----------------------|
| `learning_rate` | Learning Rate        |
| `max_depth`     | Max Depth Of Tree    |
| `n_estimators`  | Number Of Estimators |
| `objective`     | Objective Function   |
| `tree_method`   | Tree Method          |

### Model Features
The model relies on various statistical features calculated from both the time-domain and frequency-domain data of the audio samples. These features are crucial in training the model to recognize gender-specific patterns in the voice recordings.

These features include:
| Feature     | Description        |
|-------------|--------------------|
| `mean`      | Mean               |
| `std`       | Standard Deviation |
| `med`       | Median             |
| `min`       | Minimum            |
| `q25`       | 25th Percentiles   |
| `q75`       | 75th Percentiles   |
| `max`       | Maximum            |
| `skew`      | Skewness           |
| `kurt`      | Kurtosis           |
| `zeroxrate` | Zero Crossing Rate |
| `entropy`   | Entropy            |
| `sfm`       | Spectral Flatness  |
| `cent`      | Frequency Centroid |

### Model Performance
The model has achieved a promising accuracy rate of `96.70%` on the test dataset, showcasing its potential in gender recognition through voice analysis.

### PostgreSQL Integration
PostgreSQL integration enhances the project's data management capabilities. By utilizing [PostgreSQL](https://www.postgresql.org), an open-source object-relational database system, we ensure scalability, robustness, and efficient storage and retrieval of data. Utilize psycopg2, a Python driver, for seamless interaction with PostgreSQL. Store connection details in the `config/postgres.yaml` file for easy access, and perform database operations using SQL queries.

### Conclusion
This project epitomizes the application of machine learning in voice and speech analysis, showcasing the potential to discern gender through acoustic features.

### Requirements
Execute `pip install -r requirements.txt` to install the required libraries.

### Execution
Execute `audiomnist.sh` to initiate the entire pipeline.

Following arguments can be specified:
| Argument              | Description                               |
|-----------------------|-------------------------------------------|
| `-c`, `--cfg_file`    | Path to the configuration file            |
| `-y`, `--pgs_file`    | Path to the PostgreSQL configuration file |
| `-d`, `--data_prep`   | Execute the data preparation step         |
| `-f`, `--feat_eng`    | Execute the feature engineering step      |
| `-s`, `--data_split`  | Execute the data splitting step           |
| `-u`, `--model_tune`  | Execute the model tuning step             |
| `-t`, `--model_train` | Execute the model training step           |
| `-p`, `--model_pred`  | Execute the model prediction step         |
| `-q`, `--data_sql`    | Execute the data sql step                 |


### Unit Tests
Execute `python -m unittest discover test` to run all unit tests, ensuring the reliability of the code base.

### Developer
Execute `python -m pre_commit run --all-files` to ensure code quality and formatting checks.
