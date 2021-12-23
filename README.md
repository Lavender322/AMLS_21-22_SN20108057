# AMLS_21-22_SN20108057
Applied Machine Learning Systems ELEC0134 (21/22) Assignment

### Description

A brief description of the organization of the project is presented below in a logical order: 

1. Step 1 - **Feature Extraction**: feature_extractor.py
1. Step 2 - **Data Pre-processing**: data_preprocessing.py
1. Step 3 - **Model Selection**: model_selection.py
1. Step 4 - **Classification (train, validate, and test using the final models)**: task_A.py, task_B.py 
1. Step 5 - **Result Display**: result_display.py, plot_model_performance.py
1. Step 6 - **Project Execution**: main.py

### Prerequisites

In your Python 3.6 environment or machine, from the route directory of where you
cloned this project, install the required packages by running 

```python
pip install -r requirements.txt:
```

### Usage

The role of each file in this project is illustrated as follows:

* The main.py script contains the main body of this project, which is run only to train, validate, and test the optimal machine learning model selected for the specified two tasks. 
* The **task_A.py** script implements binary brain MRI tumour classification for Task A.
* The **task_B.py** script implements multiclass brain MRI tumour classification for Task B.
* The **result_display.py** script acquires the corresponding classification performance metrics on training, validation, and test datasets for a model and prints the scores to console.
* The **plot_model_performance** script plots learning curves using cross-validatioon and accuray and loss of models for each epoch.
* The **model_selection.py** script performs model selection (based on grid search hyper-parameter optimization for non-deep learning models).
* The **data_preprocessing.py** script carries out data pre-processing of the raw image data from the Kaggle MRI dataset.
* The **feature_extractor.py** script performs four feature extraction techniques for non-deep learning models.


