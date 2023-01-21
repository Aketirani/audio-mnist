# Gender Recognition By Voice & Speech Analysis

A voice recording is analyzed for many inferences like the content spoken, the emotion, gender, and identity of the speaker, and many more.

While recognizing the characteristics of the speaker from the recording, identity recognition requires a reference dataset and is limited to identifying the people in the dataset. In many cases, this level of specificity may not be needed.

Recognizing the speaker’s gender is one such use case. The model for the same can be trained to learn patterns and features specific to each gender and reproduce it for individuals who are not part of the training dataset too.

It finds applications in automatic salutations, tagging audio recording, and helping digital assistants reproduce male or female generic results.

In this repository, we will be using acoustic features extracted from a voice recording to predict the speaker’s gender.

## Dataset

[Kaggle](https://www.kaggle.com/datasets/primaryobjects/voicegender)

The dataset consists of 30.000 audio samples of spoken digits (0-9) of 60 different speakers.

There is one directory per speaker holding the audio recordings.

Additionally "audioMNIST_meta.txt" provides meta information such as gender or age of each speaker.

## Machine Learning Models
To Be Done...

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
|   ├── audio_data.csv          <-- Preprocced Audio CSV File
|   ├── final_data.csv          <-- Final Features CSV File
|
├── plots
|   ├── *.png                   <-- Audio Recording Plots
|
├── source
|   ├── config.yaml             <-- Configuration File Including Paths
|   ├── data_processing.py      <-- Data Processing Script
|   ├── feature_engineering.py  <-- Feature Engineering Script
|   ├── main.ipynb              <-- Main Script Executing Pipeline
|   ├── setup.py                <-- Setup Script
|   ├── utilites.py             <-- Utilities Script
```

## Exectuion
Run `main.ipynb` script to execute whole pipeline
