from result_display import display_metric_results_B
import plot_model_performance as plot
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import tensorflow as tf
import numpy as np


def model_B(training_images, training_labels, val_images, val_labels, test_images, test_labels, num_classes):
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
    # datagen = ImageDataGenerator(vertical_flip=True,
    #                              rotation_range=90)

    # Prepare an iterators
    # train_iterator = datagen.flow(training_images, training_labels, batch_size=32)
   
    # Fit model with generator
    # history = model.fit(train_iterator, validation_data=(val_images, val_labels), steps_per_epoch=len(train_iterator), 
    #                     epochs=20, verbose=1, callbacks=[tensorboard,checkpoint,reduce_lr])

    # Ablation study: without data augmentation
    history = model.fit(training_images, training_labels, validation_data=(val_images, val_labels), epochs=20, verbose=1, batch_size=32, 
                    callbacks=[tensorboard,checkpoint,reduce_lr])

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

    # display_metric_results_A(y_train_new, pred_train, y_val_new, pred_val, y_test_new, pred_test)
    display_metric_results_B(y_train_new, pred_train, y_val_new, pred_val, y_test_new, pred_test)
    return pred_test