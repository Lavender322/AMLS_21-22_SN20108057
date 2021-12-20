from result_display import display_metric_results_A
from result_display import display_metric_results_B
import plot_model_performance as plot
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from hypopt import GridSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from warnings import filterwarnings
from sklearn.model_selection import ShuffleSplit


# ======================================================================================================================
# SVM Classification Model
def img_SVM(training_images, training_labels, val_images, val_labels, test_images, test_labels):
    print("Fitting the classifier to the training set")

    param_grid = {'C': [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,5e1,1e2,5e2,1e3,5000],
                  'gamma': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4], } 

    gs = GridSearch(model=svm.SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid)
    clf = gs.fit(training_images, training_labels, val_images, val_labels)

    print("Best estimator found by grid search:")
    print(clf) # equivalent to print(gs.best_estimator_)

    # Plot learning curves
    fig, axes = plt.subplots(1, 1)
    title = r"Learning Curves (SVM, RBF Kernel, C=1.0, $\gamma=1e-09$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    estimator = svm.SVC(kernel='rbf', C=1, gamma=1e-9)
    plot.plot_learning_curve(estimator, title, np.concatenate((training_images, val_images)), np.concatenate((training_labels[:,0], val_labels[:,0])), axes=1, ylim=(0.75, 1.01),
                        cv=cv, n_jobs=4)
    plt.show()

    # Model evaluation
    training_pred = clf.predict(training_images) # equavalent to gs.best_estimator_.predict(training_images)
    val_pred = clf.predict(val_images)
    test_pred = clf.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    # display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# KNN Classification Model
def img_KNN(training_images, training_labels, val_images, val_labels, test_images, test_labels):
    #Create KNN object with a K coefficient
    param_grid = {'n_neighbors': list(range(1,50)),
                  'leaf_size': list(range(1,50)),
                  'p': [1,2]}
    gs = GridSearch(model=KNeighborsClassifier(), param_grid=param_grid)
    # Train the model using the training sets
    neigh = gs.fit(training_images, training_labels, val_images, val_labels) # Fit KNN model
    print("Best estimator found by grid search:")
    print(neigh)

    # Model evaluation
    training_pred = neigh.predict(training_images)
    val_pred = neigh.predict(val_images)
    test_pred = neigh.predict(test_images)
    display_metric_results_A(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    # display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    return test_pred

# ======================================================================================================================
# CNN Classification Model
def img_CNN(training_images, training_labels, val_images, val_labels, test_images, test_labels, num_classes):
    # Create a `Sequential` model
    model = Sequential([
        Conv2D(16,kernel_size=(5,5),  # num_filter, filter_size
               activation='relu',
               padding='same'),
        Dropout(0.5),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(32,(3,3),activation='relu', padding='same'),
        Dropout(0.5),
        MaxPooling2D(pool_size=(2,2)),
       
        Conv2D(64,(3,3),activation='relu', padding='same'),
        Dropout(0.5),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128,(3,3),activation='relu', padding='same'),
        Dropout(0.5),
        MaxPooling2D(pool_size=(2,2)), 
        
        Flatten(),
        Dense(1024,activation='relu'),
        Dense(num_classes,activation='softmax'),
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='logs_CNN')
    checkpoint = ModelCheckpoint("CNN_task_A.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)

    # Create generator to augment images
    datagen = ImageDataGenerator(vertical_flip=True,
                                 rotation_range=90)

    # Prepare an iterators
    train_iterator = datagen.flow(training_images, training_labels, batch_size=32)
   
    # Fit model with generator
    history = model.fit(train_iterator, validation_data=(val_images, val_labels), steps_per_epoch=len(train_iterator), 
                        epochs=20, verbose=1, callbacks=[tensorboard,checkpoint,reduce_lr])

    # Ablation study: without data augmentation
    # history = model.fit(training_images, training_labels, validation_data=(val_images, val_labels), epochs=20, verbose=1, batch_size=32, 
    #                 callbacks=[tensorboard,checkpoint,reduce_lr])

    plot.plot_accuracy_loss(history)

    # Model evaluation
    pred_train = model.predict(training_images)
    pred_train = np.argmax(pred_train, axis=1)
    y_train_new = np.argmax(training_labels, axis=1)

    pred_test = model.predict(test_images)
    pred_test = np.argmax(pred_test, axis=1)
    y_test_new = np.argmax(test_labels, axis=1)

    pred_val = model.predict(val_images)
    pred_val = np.argmax(pred_val, axis=1)
    y_val_new = np.argmax(val_labels, axis=1)

    display_metric_results_A(y_train_new, pred_train, y_val_new, pred_val, y_test_new, pred_test)
    # display_metric_results_B(y_train_new, pred_train, y_val_new, pred_val, y_test_new, pred_test)
    return pred_test

# ======================================================================================================================
# Transfer Learning with EfficientNetB0 Classification Model
def img_TL(training_images, training_labels, val_images, val_labels, test_images, test_labels, num_classes):
    effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = effnet.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=0.5)(model)
    model = tf.keras.layers.Dense(num_classes,activation='softmax')(model)
    model = tf.keras.models.Model(inputs=effnet.input, outputs = model)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='logs_ENet')
    checkpoint = ModelCheckpoint("ENet_task_A.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)

    # Create generator to augment images
    datagen = ImageDataGenerator(vertical_flip=True,
                                 rotation_range=90)

    # Prepare an iterators
    train_iterator = datagen.flow(training_images, training_labels, batch_size=32)
   
    # Fit model with generator
    history = model.fit(train_iterator, validation_data=(val_images, val_labels), steps_per_epoch=len(train_iterator), 
                        epochs=20, verbose=1, callbacks=[tensorboard,checkpoint,reduce_lr])

    # Ablation study: without data augmentation
    # history = model.fit(training_images, training_labels, validation_data=(val_images, val_labels), epochs=20, verbose=1, batch_size=32, 
    #                 callbacks=[tensorboard,checkpoint,reduce_lr])

    plot.plot_accuracy_loss(history)

    # Model evaluation  
    pred_train = model.predict(training_images)
    pred_train = np.argmax(pred_train, axis=1)
    y_train_new = np.argmax(training_labels, axis=1)

    pred_test = model.predict(test_images)
    pred_test = np.argmax(pred_test, axis=1)
    y_test_new = np.argmax(test_labels, axis=1)

    pred_val = model.predict(val_images)
    pred_val = np.argmax(pred_val, axis=1)
    y_val_new = np.argmax(val_labels, axis=1)

    display_metric_results_A(y_train_new, pred_train, y_val_new, pred_val, y_test_new, pred_test)
    # display_metric_results_B(y_train_new, pred_train, y_val_new, pred_val, y_test_new, pred_test)
    return pred_test