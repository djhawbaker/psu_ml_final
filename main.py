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
from tensorflow.keras.applications import InceptionResNetV2, VGG16, MobileNetV2, Xception, NASNetLarge
#from tensorflow.keras.Metric import reset_state
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

        :param image_size: Tuple of the input image size
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

    def reset_member_variables(self):
        """ Resets the member variables so a new model can be run without issues

        :return:
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
        self.name = None
        self.model = None
        self.predictions = None
        self.callbacks = None
        self.conf_matrix = None
        self.metrics_functions = None

        #tf.keras.metrics.Recall.reset_state()


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
        self.model.add(layers.Dense(1, activation='softmax'))
        #pre_model.trainable = False

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        #self.model.trainable = True
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=self.metrics_functions)

    def create_callbacks(self):
        """ Create the callbacks for the fit function

        :return: None
        """
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        ]

    def create_metric_functions(self):
        self.metrics_functions = [
            tf.keras.metrics.Accuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.FalseNegatives(name='false_negatives'),
            tf.keras.metrics.FalsePositives(name='false_positives'),
            tf.keras.metrics.TruePositives(name='true_positives'),
            tf.keras.metrics.TrueNegatives(name='true_negatives')
        ]

    def train(self, epochs):
        """ Train the new model on the provided data

        :param epochs: How many epochs to run
        :return: None
        """
        self.epochs = epochs
        history = self.model.fit(self.trainer, validation_data=self.validator, epochs=epochs, shuffle=False)#, callbacks=self.callbacks)
        print(f"Model: {self.name}: {history}")

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
        precision = history.history['precision']
        recall = history.history['recall']
        #f1 = history.history['f1']

        false_negatives = history.history['false_negatives']
        false_positives = history.history['false_positives']
        true_positives = history.history['true_positives']
        true_negatives = history.history['true_negatives']

        self.print_results(accuracy, precision, recall)
        self.save_results("History_", accuracy, precision, recall,
                          false_negatives, false_positives, true_negatives, true_positives)
        self.plot_results("Accuracy", "Accuracy", accuracy)
        self.plot_results("Precision", "Precision", precision)
        self.plot_results("Recall", "Recall", recall)
        #self.plot_results("F1_Score", "F1 Score", f1)
        self.plot_results("False_Negatives", "False Negatives", false_negatives)
        self.plot_results("False_Positives", "False Positives", false_positives)
        self.plot_results("True_Positives", "True Positives", true_positives)
        self.plot_results("True_Negatives", "True Negatives", true_negatives)

        # Calculated Final averaged? results
        print("Calculated Final results")
        accuracy2 = accuracy_score(labels, predictions)
        precision2 = precision_score(labels, predictions)
        recall2 = recall_score(labels, predictions)
        f12 = f1_score(labels, predictions)

        self.print_results(accuracy2, precision2, recall2, f12)
        self.save_results("Calculated_", accuracy2, precision2, recall2, f12)

    def print_results(self, accuracy, precision, recall, f1=None):
        """ Print the resulting confusion matrix

        :param accuracy:
        :param precision:
        :param recall:
        :param f1:
        :return: None
        """
        print("Model: " + self.name + " Confusion Matrix")
        print(self.conf_matrix)

        print("Accuracy: ", str(accuracy))
        print("Precision: ", str(precision))
        print("Recall: ", str(recall))
        if f1:
            print("F1 Score: ", str(f1))

    @staticmethod
    def generate_filename(base_string, ext):
        """ Generate a unique file name by incrementing the number at the end of the file
        Scans the output directory to see what the highest number is and increments by one
        If the directory 'output' doesn't exist in the local path, then it is created

        :param base_string: Base string at prepend to the output string
        :param ext: Extension of the file to create
        :return: The unique filename string
        """
        numbers = []
        if not os.path.exists('output'):
            os.makedirs('output')

        # TODO handle different length extensions
        # len_ext = len(ext) + 1
        for file in os.scandir('output'):
            if file.name.startswith(base_string):
                numbers.append(int(file.name[len(base_string):-4]))

        if len(numbers) > 0:
            filename = 'output/' + base_string + str(max(numbers) + 1) + '.' + ext
        else:
            filename = 'output/' + base_string + '1.' + ext

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
        filename = self.generate_filename(self.name + '_' + title + '_', 'png')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(y_label)
        plt.plot(plot_data)
        #plt.plot(range(len(plot_data)), plot_data)
        plt.savefig(filename)

    def save_results(self, base_filename, accuracy, precision, recall, f1=None, false_negatives=None, false_positives=None,
                     true_negatives=None, true_positives=None):
        """ Save the results to file

        :param base_filename: Name to include in the filename to signify differences
        :param accuracy: The accuracy
        :param precision: The precision
        :param recall: The recall
        :param f1: The F1 score
        :param false_negatives: The false negatives
        :param false_positives: The false positives
        :param true_negatives: The true negatives
        :param true_positives: The true positives
        :return: None
        """
        filename = self.generate_filename(self.name + '_' + base_filename, 'txt')
        with open(filename, 'w') as f:
            f.write("Results")
            f.write("\nAccuracy: " + str(accuracy))
            f.write("\nPrecision: " + str(precision))
            f.write("\nRecall: " + str(recall))

            if f1:
                f.write("\nF1 Score: " + str(f1))
            if false_negatives:
                f.write("\nFalse negatives: " + str(false_negatives))
            if false_positives:
                f.write("\nFalse positives: " + str(false_positives))
            if true_negatives:
                f.write("\nTrue negatives: " + str(true_negatives))
            if true_positives:
                f.write("\nTrue positives: " + str(true_positives))

            f.write("\nConfusion Matrix: \n")
            f.write(str(self.conf_matrix))

    def run(self, model, model_name, num_epochs):
        """ Run the whole program

        :param model: Initial model to transfer from
        :param model_name: The string name of the initial model to transfer from. Used in results output
        :param num_epochs: How many epochs to run during training
        :return: None
        """
        self.reset_member_variables()
        self.name = model_name
        self.create_metric_functions()
        self.create_model(model)
        self.create_callbacks()
        self.train(num_epochs)


if __name__ == "__main__":
    epochs = 2

    """
    # Setup model 1: InceptionResNetV2
    image_size = (299, 299)
    mm = ModelMaker(image_size)
    pre_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=image_size + (3,))
    pre_model.trainable = False
    pre_model_name = "InceptionResNetV2"
    mm.run(pre_model, pre_model_name, epochs)
    """

    # Setup model 2: VGG16
    # TODO this model asserts
    image_size = (224, 224)
    mm = ModelMaker(image_size)
    pre_model = VGG16(weights="imagenet", include_top=False, input_shape=image_size + (3,))
    pre_model.trainable = False
    pre_model_name = "VGG16"
    mm.run(pre_model, pre_model_name, epochs)

    # Setup model 3: MobileNetV2
    image_size = (224, 224)
    mm = ModelMaker(image_size)
    pre_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=image_size + (3,))
    pre_model.trainable = False
    pre_model_name = "MobileNetV2"
    mm.run(pre_model, pre_model_name, epochs)

    # Setup model 4: Xception
    image_size = (299, 299)
    mm = ModelMaker(image_size)
    pre_model = Xception(weights="imagenet", include_top=False, input_shape=image_size + (3,))
    pre_model.trainable = False
    pre_model_name = "Xception"
    mm.run(pre_model, pre_model_name, epochs)

    # Setup model 5: NASNetLarge
    image_size = (331, 331)
    mm = ModelMaker(image_size)
    pre_model = NASNetLarge(weights="imagenet", include_top=False, input_shape=image_size + (3,))
    pre_model.trainable = False
    pre_model_name = "NASNetLarge"
    mm.run(pre_model, pre_model_name, epochs)
