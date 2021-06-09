"""
ML & CVDL Final Project

Test loss and validation loss
Accuracy - TruePositives + TrueNegative / TruePositive + TrueNegative + FalsePositive + FalseNegative
Precision - TruePositives / (TruePositives + FalsePositives)
Recall - TruePositives / (TruePositives + FalseNegatives)
Processing time
F1 Score - 2 * ((precision * recall) / (precision + recall))
Sensitivity - Same as recall-> Has disease
Specificity - TrueNegative / (TrueNegative + FalsePositive) -> Doesn't have disease
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionResNetV2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
"""
Models:
1. InceptionResNetV2: https://keras.io/api/applications/inceptionresnetv2/
2. VGG16: https://keras.io/api/applications/vgg/#vgg16-function
3. MobileNetV2: https://keras.io/api/applications/mobilenet/#mobilenetv2-function
4. Xception: https://keras.io/api/applications/xception/
5. NASNetLarge: https://keras.io/api/applications/nasnet/#nasnetlarge-function

"""


class ModelMaker:
    """ Class to handle transferring the model and creating a new one

    """

    def __init__(self, image_size):
        """ Initialize the ModelMaker class

        :param trainer: The generator for training data
        :param validator: The generator for validation data
        """
        traingen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=25,
                                                                   validation_split=.15)
        testgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

        self.trainer = traingen.flow_from_directory('./data/train', class_mode="binary", classes=['COVID19', 'NORMAL'],
                                                    shuffle=False, batch_size=8, target_size=image_size,
                                                    subset="training")

        self.validator = testgen.flow_from_directory('./data/train', class_mode="binary",
                                                      classes=['COVID19', 'NORMAL'],
                                                      shuffle=False, batch_size=8, target_size=image_size,
                                                      subset="validation")

        self.tester = testgen.flow_from_directory('./data/test', class_mode="binary", classes=['COVID19', 'NORMAL'],
                                                  shuffle=False, batch_size=8, target_size=image_size)

        self.epochs = 1
        self.name = None
        self.model = None
        self.predictions = None
        self.callbacks = None
        self.conf_matrix = None
        self.metrics_functions = None

    def create_model(self, pre_model):
        """ Transfers the input model and creates a new classifier

        :param pre_model: The input model to transfer from
        :return: None
        """
        # Transfer
        self.model = models.Sequential()
        self.model.add(pre_model)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        pre_model.trainable = False

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.trainable = True
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=self.metrics_functions)

    def create_callbacks(self):
        """ Create the callbacks for the fit function

        :return: None
        """
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss')
        ]

    def create_metric_functions(self):
        self.metrics_functions = [
            tf.keras.metrics.Accuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives()
        ]

    def train(self, epochs):
        """ Train the new model on the provided data

        :param epochs: How many epochs to run
        :return: None
        """
        self.epochs = epochs
        history = self.model.fit(self.trainer, validation_data=self.validator, epochs=epochs, shuffle=False)#, callbacks=self.callbacks)
        print(f"Model 1: InceptionResNetV2: {history}")

        """
        eval_loss = self.model.evaluate(self.validator)
        print(eval_loss)

        self.predictions = np.array([])
        for x, y in self.validator:
            self.predictions = np.concatenate([self.predictions, np.argmax(self.model.predict(x), axis=-1)])
        """

        predictions = self.model.predict(x=self.tester, batch_size=32)
        # Making predictions binary based on output for initial confusion matrix
        predictions = np.array(predictions)
        predictions = np.rint(predictions)

        self.results(self.tester.classes, predictions, history)

    def results(self, labels, predictions, history):
        self.conf_matrix = confusion_matrix(labels, predictions)

        # From history
        accuracy = history.history['accuracy']
        precision = history.history['accuracy']
        recall = history.history['accuracy']
        f1 = history.history['accuracy']

        self.print_results(accuracy, precision, recall, f1)
        self.save_results("History", confusion_matrix, accuracy, precision, recall, f1)
        self.plot_results("Accuracy", "Accuracy", accuracy)
        self.plot_results("Precision", "Precision", precision)
        self.plot_results("Recall", "Recall", recall)
        self.plot_results("F1 Score", "F1 Score", f1)

        # Calculated Final results
        print("Calculated Final results")
        accuracy2 = accuracy_score(labels, predictions)
        precision2 = precision_score(labels, predictions)
        recall2 = recall_score(labels, predictions)
        f12 = f1_score(labels, predictions)

        self.print_results(accuracy2, precision2, recall2, f12)
        self.save_results("Calculated", confusion_matrix, accuracy2, precision2, recall2, f12)

    def print_results(self, accuracy, precision, recall, f1):
        """ Print the resulting confusion matrix

        :param accuracy:
        :param precision:
        :param recall:
        :param f1:
        :return: None
        """
        print("Model 1: InceptionResNetV2 Confusion Matrix")
        print(self.conf_matrix)

        print("Accuracy: ", str(accuracy))
        print("Precision: ", str(precision))
        print("Recall: ", str(recall))
        print("F1 Score: ", str(f1))

    @staticmethod
    def generate_filename(base_string):
        """ Generate a unique file name by incrementing the number at the end of the file
        Scans the output directory to see what the highest number is and increments by one
        If the directory 'output' doesn't exist in the local path, then it is created

        :param base_string: Base string at prepend to the output string
        :return: The unique filename string
        """
        numbers = []
        if not os.path.exists('output'):
            os.makedirs('output')

        for file in os.scandir('output'):
            if file.name.startswith(base_string):
                numbers.append(int(file.name[len(base_string):-4]))

        if len(numbers) > 0:
            filename = base_string + str(max(numbers) + 1) + '.png'
        else:
            filename = base_string + '1.png'

        return filename

    def plot_results(self, title, y_label, plot_data):
        """ Plot the results from throughout the training.
        Saves to file
        Call separately for each variable being plotted

        :param title: String of what is being plotted
        :param y_label: String of what should be on the vertical axis (may be the same as the title)
        :param plot_data: List of the data to plot ie. [accuracy_values, ...]
        :return: None
        """
        filename = self.generate_filename(title)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(y_label)
        plt.plot(self.epochs, plot_data)
        plt.savefig(filename)

    def save_results(self, base_filename, confusion_matrix, accuracy, precision, recall, f1):
        """ Save the results to file

        :param base_filename: Name to include in the filename to signify differences
        :param confusion_matrix: The confusion matrix
        :param accuracy: The accuracy
        :param precision: The precision
        :param recall: The recall
        :param f1: The F1 score
        :return: None
        """
        filename = self.generate_filename(base_filename)
        with open(filename, 'w') as f:
            f.write("Results")
            f.write("Accuracy: " + str(accuracy))
            f.write("Precision: " + str(precision))
            f.write("Recall: " + str(recall))
            f.write("F1 Score: " + str(f1))
            f.write("Confusion Matrix: ")
            f.write(confusion_matrix)

    def run(self, model, model_name, num_epochs):
        """ Run the whole program

        :param model: Initial model to transfer from
        :param model_name: The string name of the initial model to transfer from. Used in results output
        :param num_epochs: How many epochs to run during training
        :return: None
        """
        self.name = model_name
        self.create_metric_functions()
        self.create_model(model)
        self.create_callbacks()
        self.train(num_epochs)


if __name__ == "__main__":
    epochs = 10
    image_size = (299, 299)

    # Setup model 1: InveptionResNetV2
    pre_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=image_size + (3,))
    pre_model_name = "imagenet"

    mm = ModelMaker(image_size)
    mm.run(pre_model, pre_model_name, epochs)
