""" 
Class for Training CNNs.
Based on https://github.com/Chantalkle/DataChallenge2023/blob/main/Scripte/cnn.py (only for coin mints)
Class can be called from the console. 
"""
import argparse
import tensorflow as tf
import numpy as np
import os
import csv
import pandas
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback

class SaveTrainingDataCallback(Callback):
    def __init__(self, filename):
        super(SaveTrainingDataCallback, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            f.write(f'Epoch {epoch+1} - Loss: {logs["loss"]}, Accuracy: {logs["categorical_accuracy"]} ,ValLoss: {logs["val_loss"]}, ValAccuracy: {logs["val_categorical_accuracy"]}\n')  
            
class ConvNet():

    def __init__(self, train_path, val_path, test_path,trainingLogName, model='ResNet50V2',  img_size=(224, 224, 3), side='',subSetTestPath='None'):
        """
        train_path: Path to training data. Should be in a tree structure.
        val_path: Path to val data. Should be in a tree structure.
        test_path:  Path to test data. Should be in a tree structure.
        trainingLogName: Name of a txt file in which the training process will be logged.
        model: which ResNet model to use. Default ResNet50V2 other options ResNet101V2 and ResNet152V2 (VGG16 not working properly)
        img_size: Input size for the CNN. Default input size of ResNet (224,224,3)
        subSetTestPath: Optional. Path to another test data which will be evaluated during the training
        """
        self.side = side
        self.model_name = model + '_' + side
        self.img_size = img_size
        self.num_class = len(os.listdir(train_path))
        self.load_model(name=model)
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.subSetTestPath = subSetTestPath
        # self.classes = os.listdir(self.train_path)
        self.callbacks = []
        self.optimizer = None  
        self.augmentation = False
        self.trainingLogName = trainingLogName
        
    def load_model(self, name):
        """
        Loads a pretrained Model with a dense layer on top. Size of the dense layer equal to the number of classes.
        Other architectures can be added. See https://www.tensorflow.org/api_docs/python/tf/keras/applications/ .
        Don't forget to change the preprocessing function!! 
        """
        if name == 'ResNet50V2':
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2
            self.base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet',
                                                            input_shape=self.img_size, pooling='avg'
                                                            )
            self.preprocessing = tf.keras.applications.resnet_v2.preprocess_input
            self.model = tf.keras.Sequential()
            self.model.add(self.base_model)
            self.model.add(tf.keras.layers.Dense(
                        self.num_class, activation='softmax'))
        elif name == 'ResNet101V2':
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet101V2
            self.base_model = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet',
                                                            input_shape=self.img_size, pooling='avg'
                                                            )
            self.preprocessing = tf.keras.applications.resnet_v2.preprocess_input
            self.model = tf.keras.Sequential()
            self.model.add(self.base_model)
            self.model.add(tf.keras.layers.Dense(
                        self.num_class, activation='softmax'))
        elif name == 'ResNet152V2':
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152V2
            self.base_model = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet',
                                                            input_shape=self.img_size, pooling='avg'
                                                            )
            self.preprocessing = tf.keras.applications.resnet_v2.preprocess_input
            self.model = tf.keras.Sequential()
            self.model.add(self.base_model)
            self.model.add(tf.keras.layers.Dense(
                        self.num_class, activation='softmax'))
        
        elif name == 'VGG16':
            self.base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',include_top=False,pooling='max')
            self.base_model.trainable = False 
            # self.base_model.summary()
            self.preprocessing = tf.keras.applications.vgg16.preprocess_input

            self.model = tf.keras.Sequential([ self.base_model,
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(2048, activation='relu'),
                                        tf.keras.layers.Dense(2048, activation='relu'),
                                        # layers.Dropout(0.2),
                                        tf.keras.layers.Dense(self.num_class, activation='softmax')])
            # self.set_optimizer(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,name='SGD2')
            #                 , optimizer2=tf.keras.optimizers.SGD( learning_rate=0.00001,name='SGD2'))
            self.set_optimizer(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
                            , optimizer2=tf.keras.optimizers.Adam( learning_rate=0.00001))


    def load_coin_model(self, path ): #,mint = ''):
        """
        Loads a Tensorflow/Keras Model
        Data should be properly preprocessed!! 
        """
        self.model = tf.keras.models.load_model(path0)
        # self.mint = mint

    def freeze(self, layer_name='conv5'):
        """
        Freeze all layer until a layer named @name appears 
        """
        for layer in self.base_model.layers:
            if layer.name.startswith(layer_name):
                break
            layer.trainable = False
    def set_optimizer(self, optimizer=None, optimizer2=None):
        """ 
        Set the optimizer for training. First trained with optimizer then with second optimizer after 10 Epochs
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()
        if optimizer2 is None:
            optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.00001)

        self.optimizer = optimizer
        self.second_optimizer = optimizer2
        self.model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=[tf.keras.metrics.CategoricalAccuracy()])
        

    def set_callbacks(self, callback_list=[], set_default=False, save_training_data=False):
        """
        Adding callbacks.
        If set_default is true, early stopping and checkpoint are added to the list.
        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks list of all callbacks
        """
        if save_training_data:
            self.callbacks.append(SaveTrainingDataCallback(self.trainingLogName))

        self.callbacks += callback_list
        
        if set_default:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath='best_weights.hdf5',
                monitor='val_categorical_accuracy',
                mode='auto', save_freq='epoch', save_best_only=True)
            early = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto',
                baseline=None, restore_best_weights=False)
            self.callbacks.append(early)
            self.callbacks.append(checkpoint)

    def preprocess_images(self, x, y):
        return self.preprocessing(x), y

    def prepare(self, data):
        """
        preprocessing the images
        """
        data = data.map(self.preprocess_images)
        return data


    def load_dataset(self, batch_size, seed=None):
        """
        Function to load and preprocess the datasets.
        batch_size: Batch size for training. Test batch size must be  changed manually in the code!
        https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory for more information
        labels='inferred' is needed so that all Cnns have the same internal labels. The labels are from the structure of the 
        trainingfolder. 
        """

        self.train_set = tf.keras.utils.image_dataset_from_directory(
            self.train_path, labels='inferred', label_mode='categorical',
            batch_size=batch_size, image_size=self.img_size[:-1],  seed=123,
            shuffle=True, color_mode='rgb'
        )

        self.test_set = tf.keras.utils.image_dataset_from_directory(
            self.test_path, labels='inferred', label_mode='categorical',
            batch_size=batch_size, image_size=self.img_size[:-1],  seed=123,
            shuffle=True, color_mode='rgb'
        )
        self.val_set = tf.keras.utils.image_dataset_from_directory(
            self.val_path, labels='inferred', label_mode='categorical',  seed=123,
            image_size=self.img_size[:-1], 
            shuffle=True, color_mode='rgb'
        )

        if self.subSetTestPath != 'None': 
            self.test_subset = tf.keras.utils.image_dataset_from_directory(
                self.subSetTestPath, labels='inferred', label_mode='categorical',
                batch_size=batch_size, image_size=self.img_size[:-1],  seed=123,
                shuffle=True, color_mode='rgb'
            )
            self.test_subset = self.prepare(self.test_subset)

        self.train_set = self.prepare(self.train_set)
        self.test_set = self.prepare(self.test_set)
        self.val_set = self.prepare(self.val_set)

    def should_save_model(self,name, accuracy, dataset, filename='evaluationData.csv'):
        """ 
        Function which checks if the current trained Cnn has the highest accuracy of all 
        CNNs with the same name in evaluationData.csv 
        """
        # Loads the csv with the evaluation data from old trainings
        df = pandas.read_csv(filename)

        # Filter rows where the Name and Set match. 
        matching_rows = df[(df['Name'] == name) & (df['Set'] == dataset)]

        if matching_rows.empty:
            # No previous entries with this name and dataset, save the model
            return True
        else:
            # Check if any previous entry has higher accuracy
            max_accuracy = matching_rows['Acc'].max()
            if accuracy > max_accuracy:
                # New model has higher accuracy, save it
                return True
            else:
                # Existing model(s) have higher or equal accuracy, don't save
                return False

    def train(self):
        """
        Function for training the network.
        5 Epochs with learning rate 0.001
        30 Epochs with learning rate 0.00001
        learning rate can be altered.
        """
        if (self.augmentation):
            self.model.fit(self.augmented_dataset, epochs=5, callbacks=self.callbacks, validation_data=self.val_set )
        else:
            self.model.fit(self.train_set, epochs=5, callbacks=self.callbacks, validation_data=self.val_set )
        
        # Evaluate 
        eval_loss, eval_acc = self.model.evaluate(self.test_set )# , verbose=2)
        print("First training finished.")
        print(f"{self.trainingLogName} loss: {eval_loss}, acc: {eval_acc}")

        # second training
        self.model.compile(optimizer=self.second_optimizer,
                            loss=tf.keras.losses.CategoricalCrossentropy(),
                                #from_logits=True),
                            metrics=[tf.keras.metrics.CategoricalAccuracy()])
        # check if augmented data should be used.
        if (self.augmentation):
            self.model.fit(self.augmented_dataset, callbacks=self.callbacks,epochs=30, validation_data=self.val_set)
        else:
            self.model.fit(self.train_set, callbacks=self.callbacks, epochs=30, validation_data=self.val_set)
        # Evaluate training
        eval_loss, eval_acc = self.model.evaluate(self.test_set)
        print(f"{self.trainingLogName} loss: {eval_loss}, acc: {eval_acc}")
        # Check if the models is best perfoming one
        csvFile = 'evaluationData.csv'
        saveModel = self.should_save_model(self.trainingLogName,eval_acc,'all types',csvFile)
        # Save training evaluation into the csv file.
        data = {'Name': [self.trainingLogName], 'Set': ['all types'], 'Loss': [eval_loss], 'Acc': [eval_acc]}
        pandaDataframe = pandas.DataFrame(data)
        pandaDataframe.to_csv(csvFile, index=False, header=False, mode='a')
        # If a second testset is given, evaluate the model over it.
        if self.subSetTestPath != 'None': 
            eval_loss, eval_acc = self.model.evaluate(self.test_subset )# , verbose=2)
            safeSubSetModel = self.should_save_model(self.trainingLogName,eval_acc,'subset types',csvFile)
            # save best model on this test set if needed.
            if safeSubSetModel:
                save_name = f"{self.trainingLogName}_bestSubSetAcc.h5"
                self.model.save(save_name)
            # Save training evaluation into the csv file.
            data = {'Name': [self.trainingLogName], 'Set': ['subset types'],'Loss': [eval_loss], 'Acc': [eval_acc]}
            pandaDataframe = pandas.DataFrame(data)
            csvFile = 'evaluationData.csv'
            pandaDataframe.to_csv(csvFile, index=False, header=False, mode='a')     

        # Save model 
        if saveModel:
            save_name = f"{self.trainingLogName}.h5"
            self.model.save(save_name)

        print("sucess")

    def apply_dataaugmentation(self,flip="horizontal",rot=0.2):
        """
        Function top apply data augmentation to the trainging data
        """
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(rot),
        ])
        #tf.keras.layers.RandomFlip(flip),
        self.augmented_dataset = self.train_set.map(lambda x, y: (self.data_augmentation(x, training=True), y))
        self.augmentation = True

    



def main(side, trainPath, valPath, testPath, batchSize, model, modelSaveName, freezeLayer, subSetTestPath, augmentation):
    """ 
    This class can be called from the console. All arguments have to be provided. subSetTestPath and augmentation are optional.
    """
    coinNet = ConvNet(train_path=trainPath,
                      val_path=valPath,
                      test_path=testPath,
                      trainingLogName=modelSaveName,
                      model=model,
                      side=side,
                      subSetTestPath=subSetTestPath)
    
    if freezeLayer != '' and model != 'VGG16':
        coinNet.freeze(freezeLayer)
        
    coinNet.set_callbacks([], True, True)
    coinNet.load_dataset(batch_size=batchSize)
    if augmentation:
        coinNet.apply_dataaugmentation()
    if  model != 'VGG16':
        coinNet.set_optimizer()
    coinNet.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for cnn")

    parser.add_argument("side", help="Type of coins to train the model on (e.g., 'Obv', 'Rev').")
    parser.add_argument("trainPath", help="Path to the directory containing training images.")
    parser.add_argument("valPath", help="Path to the directory containing validation images.")
    parser.add_argument("testPath", help="Path to the directory containing test images.")
    parser.add_argument("batchSize",  type=int,  help="Batch size to use for training the model.")
    parser.add_argument("model", help="Name of the CNN model architecture to use (e.g., 'ResNet50V2').")

    parser.add_argument("modelSaveName", help="Name for saving the trained model file and log files.")
    parser.add_argument("freezeLayer", help="Name of the layer of the model to freeze during training. ")
    parser.add_argument("subSetTestPath", help="Path to the directory containing a subset of the test images. Optional")
    parser.add_argument("--augmentation", action="store_true", help="True for augmentation. Optional. If --augmentation is not declared it defaults to False.")
    
    args = parser.parse_args()
    main(args.side, args.trainPath, args.valPath, args.testPath, args.batchSize, args.model, args.modelSaveName, args.freezeLayer, args.subSetTestPath, args.augmentation)

