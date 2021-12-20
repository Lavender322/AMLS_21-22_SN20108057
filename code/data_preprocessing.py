import feature_extractor as fe
import os  # needed navigate the system to get the input data
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from numpy import load  # load numpy array from npy file
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Data Preparation
# folderPath = os.path.join("dataset/", "image")
# images = ['IMAGE_0003.jpg', 'IMAGE_0000.jpg', 'IMAGE_0009.jpg', 'IMAGE_0001.jpg']
# labels = ['a) glioma_tumor', 'b) meningioma_tumor', 'c) pituitary_tumor', 'd) no_tumor']

# fig = plt.figure(figsize=(10,10))
# for i in range(4):
#     img = cv2.imread(os.path.join(folderPath, images[i]))
#     ax = fig.add_subplot(2, 2, i+1)
#     ax.imshow(img)
#     ax.set_title(labels[i], size=18)
# plt.savefig("dataset_examples.jpg", dpi=150)
# plt.show()


# Create Task A label
# df = pd.read_csv("dataset/label.csv")
# df.loc[df["label"]=="meningioma_tumor", "label"] = "tumor"
# df.loc[df["label"]=="glioma_tumor", "label"] = "tumor"
# df.loc[df["label"]=="pituitary_tumor", "label"] = "tumor"
# df.to_csv("drive/MyDrive/dataset/label_task_A.csv", index=False)

def last_4chars(x):
        return(x[-8:-4])


def data_preprocessing_task_a():
    labels_task_A = ['no_tumor','tumor']

    # Load images - X
    X = []
    X_ndl = []
    image_size = 256

    folderPath = os.path.join("dataset/", "image")
    images = os.listdir(folderPath)

    # For non-deep learning models
    for image in sorted(images, key = last_4chars):  
        img = cv2.imread(os.path.join(folderPath, image), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        X_ndl.append(img)
    X_ndl = np.array(X_ndl)

    # For deep learning models
    for image in sorted(images, key = last_4chars):  
        img = cv2.imread(os.path.join(folderPath, image))
        img = cv2.resize(img, (image_size, image_size))
        X.append(img)
    X = np.array(X)

    # Load labels - y
    df = pd.read_csv("dataset/label_task_A.csv")
    y = df.label.to_list()
    y = np.array(y)

    # Feature Extraction
    # fe.extract_features(X_ndl)

    # Load features
    X_ndl = load("features.npy")

    # For deep learning models
    # Create test and train sets from one dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)
    # Create a validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1) # 0.1 x 0.9 = 0.09

    # For non-deep learning models
    # Create test and train sets from one dataset
    X_train_ndl, X_test_ndl, y_train_ndl, y_test_ndl = train_test_split(X_ndl, y, test_size=0.1, random_state=20)
    # Create a validation set
    X_train_ndl, X_val_ndl, y_train_ndl, y_val_ndl = train_test_split(X_train_ndl, y_train_ndl, test_size=0.1, random_state=1) # 0.1 x 0.9 = 0.09

    # Normalize images
    X_train = np.array(X_train, dtype="float") / 255.0
    X_val = np.array(X_val, dtype="float") / 255.0
    X_test = np.array(X_test, dtype="float") / 255.0
    X_train_ndl = np.array(X_train_ndl, dtype="float") / 255.0
    X_val_ndl = np.array(X_val_ndl, dtype="float") / 255.0
    X_test_ndl = np.array(X_test_ndl, dtype="float") / 255.0

    # One hot encoding
    y_train_new = []
    for i in y_train:
        y_train_new.append(labels_task_A.index(i))
    y_train = y_train_new
    y_train = tf.keras.utils.to_categorical(y_train)

    y_val_new = []
    for i in y_val:
        y_val_new.append(labels_task_A.index(i))
    y_val = y_val_new
    y_val = tf.keras.utils.to_categorical(y_val)

    y_test_new = []
    for i in y_test:
        y_test_new.append(labels_task_A.index(i))
    y_test = y_test_new
    y_test = tf.keras.utils.to_categorical(y_test)

    y_train_new = []
    for i in y_train_ndl:
        y_train_new.append(labels_task_A.index(i))
    y_train_ndl = y_train_new
    y_train_ndl = tf.keras.utils.to_categorical(y_train_ndl)

    y_val_new = []
    for i in y_val_ndl:
        y_val_new.append(labels_task_A.index(i))
    y_val_ndl = y_val_new
    y_val_ndl = tf.keras.utils.to_categorical(y_val_ndl)

    y_test_new = []
    for i in y_test_ndl:
        y_test_new.append(labels_task_A.index(i))
    y_test_ndl = y_test_new
    y_test_ndl = tf.keras.utils.to_categorical(y_test_ndl)

    return X_train, y_train, X_val, y_val, X_test, y_test, \
           X_train_ndl, y_train_ndl, X_val_ndl, y_val_ndl, X_test_ndl, y_test_ndl 

# ======================================================================================================================
def data_preprocessing_task_b():
    labels_task_B = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

    # Load images - X
    X = []
    X_ndl = []
    image_size = 256

    folderPath = os.path.join("dataset/", "image")
    images = os.listdir(folderPath)

    # For non-deep learning models
    for image in sorted(images, key = last_4chars):  
        img = cv2.imread(os.path.join(folderPath, image), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        X_ndl.append(img)
    X_ndl = np.array(X_ndl)

    # For deep learning models
    for image in sorted(images, key = last_4chars):  
        img = cv2.imread(os.path.join(folderPath, image))
        img = cv2.resize(img, (image_size, image_size))
        X.append(img)
    X = np.array(X)

    # Load labels - y
    df = pd.read_csv("dataset/label.csv")
    y = df.label.to_list()
    y = np.array(y)

    # Load features
    X_ndl = load("features.npy")

    # For deep learning models
    # Create test and train sets from one dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)
    # Create a validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1) # 0.1 x 0.9 = 0.09

    # For non-deep learning models
    # Create test and train sets from one dataset
    X_train_ndl, X_test_ndl, y_train_ndl, y_test_ndl = train_test_split(X_ndl, y, test_size=0.1, random_state=20)
    # Create a validation set
    X_train_ndl, X_val_ndl, y_train_ndl, y_val_ndl = train_test_split(X_train_ndl, y_train_ndl, test_size=0.1, random_state=1) # 0.1 x 0.9 = 0.09

    # Normalize images
    X_train = np.array(X_train, dtype="float") / 255.0
    X_val = np.array(X_val, dtype="float") / 255.0
    X_test = np.array(X_test, dtype="float") / 255.0
    X_train_ndl = np.array(X_train_ndl, dtype="float") / 255.0
    X_val_ndl = np.array(X_val_ndl, dtype="float") / 255.0
    X_test_ndl = np.array(X_test_ndl, dtype="float") / 255.0

    # For deep learning models: one hot encoding
    y_train_new = []
    for i in y_train:
        y_train_new.append(labels_task_B.index(i))
    y_train = y_train_new
    y_train = tf.keras.utils.to_categorical(y_train)

    y_val_new = []
    for i in y_val:
        y_val_new.append(labels_task_B.index(i))
    y_val = y_val_new
    y_val = tf.keras.utils.to_categorical(y_val)

    y_test_new = []
    for i in y_test:
        y_test_new.append(labels_task_B.index(i))
    y_test = y_test_new
    y_test = tf.keras.utils.to_categorical(y_test)

    # For non-deep learning models: Ondinal encoding
    y_train_new = []
    for i in y_train_ndl:
        y_train_new.append(labels_task_B.index(i))
    y_train_ndl = y_train_new

    y_val_new = []
    for i in y_val_ndl:
        y_val_new.append(labels_task_B.index(i))
    y_val_ndl = y_val_new

    y_test_new = []
    for i in y_test_ndl:
        y_test_new.append(labels_task_B.index(i))
    y_test_ndl = y_test_new
    return X_train, y_train, X_val, y_val, X_test, y_test, \
           X_train_ndl, y_train_ndl, X_val_ndl, y_val_ndl, X_test_ndl, y_test_ndl