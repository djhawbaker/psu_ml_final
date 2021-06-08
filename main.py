"""
ML & CVDL Final Project

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionResNetV2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
"""

"""

if __name__ == "__main__":

    traingen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=25, validation_split=.15)

    # Create iterators for train, validation and test set
    trainer = traingen.flow_from_directory('./data/train', class_mode="binary", classes=['COVID19', 'NORMAL'],
                                           shuffle=False, batch_size=8, target_size=(299, 299), subset="training")

    validator = traingen.flow_from_directory('./data/train', class_mode="binary", classes=['COVID19', 'NORMAL'],
                                             shuffle=False, batch_size=8, target_size=(299, 299), subset="validation")


    # Load and preprocess images
    """
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                                                                        batch_size=8,
                                                                        image_size=(400, 400),
                                                                        seed=123,
                                                                        shuffle=False,
                                                                        subset='validation',
                                                                        validation_split=0.15)

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                                                                       batch_size=8,
                                                                       image_size=(400, 400),
                                                                       seed=123,
                                                                       shuffle=False,
                                                                       subset='validation',
                                                                       validation_split=0.15)

    """
    # Setup model
    pre_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
    #class_names = trainer.class_names
    labels = np.concatenate([y for x, y in validator], axis=0)

    # Transfer
    model = models.Sequential()
    model.add(pre_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    pre_model.trainable = False


    # Train
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.trainable = True
    model.compile(loss='binary_crossentropy', optimizer=opt)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss')
    ]
    history = model.fit(trainer, validation_data=validator, epochs=10, shuffle=False)#, callbacks=my_callbacks)
    print(history)

    eval_loss = model.evaluate(validator)
    print(eval_loss)

    predictions = np.array([])
    for x, y in validator:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])

    print("part ii")
    print(confusion_matrix(labels, predictions))
