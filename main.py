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
Models:
1. InceptionResNetV2: https://keras.io/api/applications/inceptionresnetv2/
2. VGG16: https://keras.io/api/applications/vgg/#vgg16-function
3. MobileNetV2: https://keras.io/api/applications/mobilenet/#mobilenetv2-function
4. Xception: https://keras.io/api/applications/xception/
5. NASNetLarge: https://keras.io/api/applications/nasnet/#nasnetlarge-function

"""

if __name__ == "__main__":

    traingen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=25, validation_split=.15)

    # Create iterators for train, validation and test set
    # Load and preprocess images
    trainer = traingen.flow_from_directory('./data/train', class_mode="binary", classes=['COVID19', 'NORMAL'],
                                           shuffle=False, batch_size=8, target_size=(299, 299), subset="training")

    validator = traingen.flow_from_directory('./data/train', class_mode="binary", classes=['COVID19', 'NORMAL'],
                                             shuffle=False, batch_size=8, target_size=(299, 299), subset="validation")

    # Setup model 1: InveptionResNetV2
    pre_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

    class ModelMaker:
        def __init__:
            pass


    # Transfer
    model1 = models.Sequential()
    model1.add(pre_model)
    model1.add(layers.Flatten())
    model1.add(layers.Dense(256, activation='relu'))
    model1.add(layers.Dense(1, activation='sigmoid'))
    pre_model.trainable = False

    # Train
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model1.trainable = True
    model1.compile(loss='binary_crossentropy', optimizer=opt)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss')
    ]
    history1 = model1.fit(trainer, validation_data=validator, epochs=10, shuffle=False)#, callbacks=my_callbacks)
    print(f"Model 1: InceptionResNetV2: {history1}")

    eval_loss = model1.evaluate(validator)
    print(eval_loss)

    predictions = np.array([])
    for x, y in validator:
        predictions = np.concatenate([predictions, np.argmax(model1.predict(x), axis=-1)])

    print("Model 1: InceptionResNetV2 Confusion Matrix")
    print(confusion_matrix(validator.labels, predictions))

