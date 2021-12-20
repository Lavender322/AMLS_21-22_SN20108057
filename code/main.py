import data_preprocessing as dp
import plot_model_performance as plot
from task_A import model_A
from task_B import model_B
from model_selection import img_SVM
from model_selection import img_KNN
from model_selection import img_CNN
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
import plot_model_performance as lc
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# ======================================================================================================================
# Data preprocessing
# ======
# Task A
# ======
X_train, y_train, X_val, y_val, X_test, y_test, \
    X_train_ndl, y_train_ndl, X_val_ndl, y_val_ndl, X_test_ndl, y_test_ndl = dp.data_preprocessing_task_a()

# ======
# Task B
# ======
X_train_task_B, y_train_task_B, X_val_task_B, y_val_task_B, X_test_task_B, y_test_task_B, \
    X_train_ndl_task_B, y_train_ndl_task_B, X_val_ndl_task_B, y_val_ndl_task_B, X_test_ndl_task_B, y_test_ndl_task_B = dp.data_preprocessing_task_b()

# ======================================================================================================================
# Train, validate and test the model for each task
# =======
# Task A1
# =======
# Train model based on the training set
# Fine-tune the model based on the validation set
# Generalise the model based on the test set
# Compute and print the final results to the console
acc_A_test = model_A(X_train, y_train, X_val, y_val, X_test, y_test, 2)

# =======
# Task A2
# =======
# Train model based on the training set
# Fine-tune the model based on the validation set
# Generalise the model based on the test set
# Compute and print the final results to the console
acc_B_test = model_B(X_train_task_B, y_train_task_B, X_val_task_B, y_val_task_B, X_test_task_B, y_test_task_B, 4)

# ======================================================================================================================
# Grid Search
# ======
# Task A
# ======
#pred = img_SVM(X_train_ndl, y_train_ndl[:,0], X_val_ndl, y_val_ndl[:,0], X_test_ndl, y_test_ndl[:,0])
#pred = img_KNN(X_train_ndl, y_train_ndl[:,0], X_val_ndl, y_val_ndl[:,0], X_test_ndl, y_test_ndl[:,0])
#pred = img_CNN(X_train, y_train, X_val, y_val, X_test, y_test, 2)
#pred = img_TL(X_train, y_train, X_val, y_val, X_test, y_test, 2)
# ======
# Task B
# ======
#pred = img_SVM(X_train_ndl_task_B, y_train_ndl_task_B, X_val_ndl_task_B, y_val_ndl_task_B, X_test_ndl_task_B, y_test_ndl_task_B)
#pred = img_KNN(X_train_ndl_task_B, y_train_ndl_task_B, X_val_ndl_task_B, y_val_ndl_task_B, X_test_ndl_task_B, y_test_ndl_task_B)
#pred = img_CNN(X_train_task_B, y_train_task_B, X_val_task_B, y_val_task_B, X_test_task_B, y_test_task_B, 4)
#pred = img_TL(X_train_task_B, y_train_task_B, X_val_task_B, y_val_task_B, X_test_task_B, y_test_task_B, 4)