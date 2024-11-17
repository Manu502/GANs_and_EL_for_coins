"""
Python Class to create predictions for the Ensemble Learning or for creating  a Confusion Matrix and a file with the wrong predictions.
Can be called from the Console.
Class only works for ResNet Models from Tensorflow. If you want to get the predictions from another modell add the right preprocessing function.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import csv
import argparse
import pickle
import pandas as pd
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model_path, val_path, batch_size=32, img_size=(224, 224, 3)):
        self.val_path = val_path # path to evaluation data.
        self.img_size = img_size 
        self.batch_size = batch_size
        self.model_path = model_path # path to cnn.

    def load_coin_model(self): #,mint = ''):
        """
        Loads a Tensorflow/Keras Model
        Data should be properly preprocessed!! 
        """
        self.model = tf.keras.models.load_model(self.model_path)
        # self.mint = mint

    def preprocess_images(self, x, y):
        return preprocess_input(x), y

    def prepare(self, data):
        data = data.map(self.preprocess_images)
        return data
    
    # Function to extract file paths
    # If you using tf version > 2.4 you can get the filnames via file_names from self.val_set
    def extract_file_paths(self, file_path):
        return file_path

    def evaluate_model(self):
        # Load validation data. Shuffle must be False to get the right file names.
        self.val_set = tf.keras.utils.image_dataset_from_directory(
            self.val_path, labels='inferred', label_mode='categorical',
            batch_size=self.batch_size, image_size=self.img_size[:-1], shuffle=False, color_mode='rgb'
        )
        self.class_names = self.val_set.class_names
        # Prepare validation data
        self.val_set = self.prepare(self.val_set)
        self.load_coin_model()
        # Make predictions
        self.predictions = self.model.predict(self.val_set)

        # # Get true labels
        self.true_labels = np.concatenate([y for x, y in self.val_set], axis=0)
        self.true_labels = np.argmax(self.true_labels, axis=1)

        # # Get predicted labels
        self.predicted_labels = np.argmax(self.predictions, axis=1)
        # getting the top 5 labels. fliplr used so that the top1 class is in the first position of the array
        self.predicted_top5_labels = np.fliplr(np.argsort(self.predictions, axis=1)[:,-5:])

        # # Create confusion matrix
        # cm = confusion_matrix(true_labels, predicted_labels)

        # If you using tf version > 2.4 you can get the filnames via file_names from self.val_set
        # Getting file names. Shuffle must be False to get the right file names.
        data_dir = tf.data.Dataset.list_files(self.val_path + '*/*', shuffle=False)
        # Map the function to the dataset to extract file paths
        file_paths_dataset = data_dir.map(self.extract_file_paths)
        # Convert the dataset to a list
        self.file_paths_list = list(file_paths_dataset.as_numpy_iterator())

        eval_loss, eval_acc = self.model.evaluate(self.val_set)
        # print(f"{self.trainingLogName} loss: {eval_loss}, acc: {eval_acc}")
        model_name = Path(self.model_path).stem
        set_name =  Path(self.val_path).stem
        csvFile = 'evaluationData.csv'
        # Save data
        data = {'Name': [model_name], 'Set': [set_name], 'Loss': [eval_loss], 'Acc': [eval_acc]}
        pandaDataframe = pd.DataFrame(data)
        pandaDataframe.to_csv(csvFile, index=False, header=False, mode='a')

        # return cm

    def get_predictions(self, path):
        """ 
        Function to get the prediction of data and save it as an numpy array
        """
        self.batch_set = tf.keras.utils.image_dataset_from_directory(
                path, labels='inferred', label_mode='categorical',
                image_size=self.img_size[:-1],
                batch_size=self.batch_size,shuffle=False, color_mode='rgb'
        )
        self.class_names = self.batch_set.class_names
        # Prepare validation data
        self.batch_set = self.prepare(self.batch_set)
        self.load_coin_model()

        results = []
        # Iterate over the data generator and perform predictions on each batch
        for batch_images, batch_labels in self.batch_set:
            predictions = self.model.predict_on_batch(batch_images)
            
            # Store the activations and labels in the results list
            for prediction, label in zip(predictions, batch_labels):
                results.append([prediction.tolist(), label])

        data_array = np.array(results)
        model_name =  Path(self.model_path).stem
        # Save as a NumPy binary file
        save_name = path.split('/')[-1] + '_predictions_' +  model_name  +'.npy'
        np.save(save_name, data_array)
        # return results
        class_names_dict = {index: class_name for index, class_name in enumerate(self.class_names)}
        # Save the dictionary to disk as a text file
        save_name_pkl = 'class_names_dict' + model_name   + '.pkl'
        # Save the dictionary to a file
        with open(save_name_pkl, 'wb') as f:
            pickle.dump(class_names_dict, f)
        output_file = "namesForJson.txt"

        # Append the strings to the text file
        with open(output_file, "a") as file:
            file.write(model_name + ":" + save_name + "\n")
    
    def save_plot(self):
        translated_true_labels = [self.class_names[label] for label in self.true_labels]
        translated_pred_labels = [self.class_names[label] for label in self.predicted_labels]
        # create a subset of all labels with only the predicted lables and true lables
        num_true_labels = len(translated_true_labels)
        self.sub_labels = np.unique(np.hstack((self.true_labels,self.predicted_labels)))
        # create a list with the class_names which are used
        only_used_labels = list(map(int, translated_true_labels))
        only_used_labels = np.sort(np.unique(only_used_labels))
        only_used_labels = list(map(str, only_used_labels))
        for label in translated_pred_labels:
            if label not in only_used_labels:
                only_used_labels.append(label)
        self.confusionMatrix = cm = confusion_matrix(translated_true_labels, translated_pred_labels, labels=only_used_labels, normalize='true')
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=only_used_labels)
        fig, ax = plt.subplots(figsize=(30, 30))
        cm_display.plot(ax=ax, xticks_rotation=90, include_values=False)
        # ax.set_yticks(np.arange(num_true_labels))
        ax.tick_params(axis='both', which='major')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        save_name = Path(self.model_path).stem + "_" +  Path(self.val_path).stem + '_confusion_matrix.png'
        plt.savefig(save_name)  # Save the plot as an image file

    def save_predictions_to_csv(self):
        """ 
        This functions only saves the wrong predictions.
        """
        filename = Path(self.model_path).stem + "_" +  Path(self.val_path).stem +  '_wrong_predictions.csv'
        translated_true_labels = []
        translated_pred_labels = []
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Coin', 'true_label', 'top1', 'top2', 'top3', 'top4', 'top5'])
            
            for i in range(len(self.true_labels)):
                true_label = self.true_labels[i]
                top5_labels = self.predicted_top5_labels[i]
                file_name = self.file_paths_list[i]
                
                true_label_name = self.class_names[true_label]
                translated_true_labels.append(true_label_name)

                top5_label_names = [self.class_names[label] for label in top5_labels]
                translated_pred_labels.append(top5_label_names[0])

                if true_label != top5_labels[0]:
                    writer.writerow([file_name, true_label_name] + top5_label_names)
        # workaround to show only the used labels in the classification report.
        only_used_labels =  np.unique(np.hstack((translated_true_labels,translated_pred_labels)))
        only_used_labels = list(map(int, only_used_labels))
        only_used_labels = np.sort(only_used_labels)
        only_used_labels = list(map(str, only_used_labels))
        save_name = Path(self.model_path).stem + "_" +  Path(self.val_path).stem
        clf_report = pd.DataFrame(
            classification_report(translated_true_labels, translated_pred_labels, labels=only_used_labels, target_names=only_used_labels, output_dict=True, zero_division=0.0))
        clf_report.to_csv("classification_report_" + save_name + "_"  + ".csv")

            

def main(model_path, val_path,batch_size,eval):
    if eval:
        evaluator = ModelEvaluator(model_path=model_path,
                        val_path=val_path)
        evaluator.evaluate_model()
        evaluator.save_plot()
        evaluator.save_predictions_to_csv()
    else:
        evaluator = ModelEvaluator( model_path=model_path,
                                    val_path=val_path,
                                    batch_size=batch_size)
        evaluator.get_predictions(val_path)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for cnn")

    parser.add_argument("model_Path", help="Path to CNN Model.")
    parser.add_argument("val_path", help="Path to the dataset to evaluate.")
    parser.add_argument("batch_size",  type=int, help="batch_size.")
    # parser.add_argument("save_path", help="save_path.")
    parser.add_argument("--eval", action="store_true", help="True for generating evaluation Data like a confusion matrix. If False the predictions are generated.")
    args = parser.parse_args()
    main( args.model_Path, args.val_path,args.batch_size,args.eval)