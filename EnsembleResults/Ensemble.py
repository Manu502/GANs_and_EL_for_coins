""""
Class for the Ensemble Learning.
Based on https://github.com/Chantalkle/DataChallenge2023/tree/main/Scripte (only for coin mints)
New class for voting and stacking (here used for coin types, but could also be used for mints).
User is able to decide the ensemble (different JSON files) and 
combination function (different voting and stacking functions).
Conditions: 
- Networks must be trained with the same trainset (same order, same types, same coin and same image amount)
- Generated pickle file is for all networks the same (dictionary for the right coding of the types)
- Predictions for the validation- and testset are made with the ModelEvaluator class befor using the Ensemble class
- Every network must make the predictions on the same sets (same order, same types, same coin and same image amount)
"""

import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import pickle
import argparse
import json
from collections import defaultdict
from pathlib import Path


# Check Versions
"""print("numpy: ", np.__version__)
print("matplotlib: ", matplotlib.__version__)
print("sklearn: ", sklearn.__version__)
print("panda: ", pd.__version__)
print("pickle: ", pickle.format_version)
print("argparse: ", argparse.__version__)
print("json: ", json.__version__)"""

class Ensemble():

    def __init__(self, json_file, dict_types, top_x, plot):
        """
        json_file: path to jason file with all cnns and prediction paths
        dict_types: path to pickle with types (all the same, choose one)
        top_x: number for top predictions (only for some voting functions)
        """
        self.networks = []
        self.test_predictions_path = []
        self.val_predictions_path = []
        self.test_predictions_array = []
        self.val_predictions_array = []
        # self.weights = []
        with open(dict_types, 'rb') as f:
            self.dict_types = pickle.load(f)
        self.top_x = int(top_x)
        self.json_file = Path(json_file).stem
        self.load_predictions_array_from_json(json_file)
        self.plot = plot

    def load_predictions_array_from_json(self, json_file):
        """
        json_file: path to jason file with all cnns and prediction paths
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            # Save name of base learner
            self.networks.append(key)
            values = [val for val in value.split(',')]
            # extract preds path  from base learener
            # eval path
            self.test_predictions_path.append(values[0])
            # test path
            self.val_predictions_path.append(values[1])
            # weights
            # self.weights.append(int(values[2]))

    def load_test_predictions_array(self):
        """
        Array for the predictions of the testset.
        """
        for i in range(len(self.test_predictions_path)):
            pred = np.load(self.test_predictions_path[i])
            model = self.networks[i]
            self.test_predictions_array.append((model, pred))
        self.num_of_test_elements = len(self.test_predictions_array[0][1])

    def load_val_predictions_array(self):
        """
        Array for the predictions of the validation.
        """
        for i in range(len(self.val_predictions_path)):
            pred = np.load(self.val_predictions_path[i])
            model = self.networks[i]
            self.val_predictions_array.append((model, pred))

    def top_x_predictions(self, image_model_prediction):
        """
        image_model_prediction: one image prediction of all image predictions of the given model
        from predictions_array: predictions_array[model][all_predictions][one_image][image_prediction]
        """
        top_x_indices = np.argsort(image_model_prediction)[-self.top_x:]
        # print(top_x_indices)
        savings = []

        for index in top_x_indices:
            class_name = self.dict_types[index]
            probability = image_model_prediction[index]
            savings.append((class_name, probability))

        return savings

    def top_1_predictions(self, image_model_prediction):
        """
        image_model_prediction: one image prediction of all image predictions of the given model
        from predictions_array: predictions_array[model][all_predictions][one_image][image_prediction]
        """
        top_x_indices = np.argsort(image_model_prediction)[-self.top_x:]
        # print(top_x_indices)
        savings = []

        for index in top_x_indices:
            class_name = self.dict_types[index]
            probability = image_model_prediction[index]
            savings.append(class_name)
            savings.append(probability)

        return savings

    def stack_val_predictions(self):
        """
        Predictions of the validationset are used for the meta model training.
        """
        all_preds = []
        for model in self.val_predictions_array:
            x_pred = []
            for i in range(len(model[1])):
                x_pred.append(model[1][i][0])
            all_preds.append(x_pred)
        all_preds = tuple(all_preds)
        stack = np.column_stack(all_preds)
        return stack

    def stack_test_predictions(self):
        """
        Predictions of the testset are used for the evaluation.
        """
        all_preds = []
        for model in self.test_predictions_array:
            x_pred = []
            for i in range(len(model[1])):
                x_pred.append(model[1][i][0])
            all_preds.append(x_pred)
        all_preds = tuple(all_preds)
        stack = np.column_stack(all_preds)
        return stack

    def stack_predictons_labels(self, meta_model, stacked_pred, stackingRF_predictions):
        """
        Checks if a prediction is made.
        If no prediction is made, take the maximal probability prediction.
        Save and return the labels.
        """
        pred_meta_labels = []
        index = -1

        # print(stackingRF_predictions)
        for pred in stackingRF_predictions:
            index = index + 1
            # print("Index of Prediction: ", index, "\n", "Prediction: ", pred)
            if 1 in pred.tolist():
                pred_meta_labels.append(self.dict_types[pred.tolist().index(1)])
            else:
                prob_pred = meta_model.predict_proba([stacked_pred[index]])
                max_value = 0
                i = -1
                label_index = 0
                for array in prob_pred:
                    i = i + 1
                    if array[0][1] >= max_value:
                        max_value = array[0][1]
                        label_index = i
                    else:
                        continue
                pred_meta_labels.append(self.dict_types[label_index])
        pred_meta_labels = np.array(pred_meta_labels)
        # print(len(pred_meta_labels), pred_meta_labels)
        return pred_meta_labels

    def build_label_array(self, pred_array):
        """
        Array for the true labels of the predicted images.
        """
        true_labels = []
        true_labels_name = []
        for i in range(len(pred_array[0][1])):
            true_labels.append(pred_array[0][1][i][1])
            true_labels_name.append(self.dict_types[list(pred_array[0][1][i][1]).index(1)])
        return [true_labels, true_labels_name]

    def train_meta_model(self, train_stack, train_labels, meta_model_type):
        """
        train_stack: the stacked predictions needed for the training
        train_labels: the true labels of the predictions
        meta_model_type: RF for random forrest and LR for logistic regression
        """
        train_labels = np.array(train_labels)
        if meta_model_type == "RF":
            # Change n_estimators
            # Tested with AllTest13Preds: 100 (44/56) = 800 (44/56) = 1000 (44/56) = 1600 (44/56) > 90 (43/56) = 400 (43/56) =  1200 (43/56) = 1400 (43/56) > 600 (42/56) = 1800 (42/56) = 2000 (42/56) = 2070 (42/56) 
            # Test with AllTest13Pred with 1000 estimators entropy criterion: (32/56)
            # Test with AllTest13Pred with 1000 estimators log_loss criterion: (30/56)
            # Test with AllTest13Pred with 1000 estimators max_depth: 25 (22/56) < 50 (27/56) < 100 (35/56) < 150 (42/56) = 200 (42/56) = 207 (42/56) < 225 (43/56) < None (44/56)
            meta_model_RF = RandomForestClassifier(n_estimators=1000)
            training_RF = meta_model_RF.fit(train_stack, train_labels)
            # save
            joblib.dump(training_RF, "MetaModel_RF_"+ self.json_file + "_Top" + str(self.top_x) + ".joblib")
            return training_RF
        elif meta_model_type == "LR":
            # penalty: Specify the norm of the penalty:
                # None,'l2', 'l1', 'elasticnet'
            # solver: For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss.
            # Test for AllBothTestPreds:
            # solver and penalty:
                # newton-cg: l2 (894/1040) > None (854/1040)
                # sag: l2 (894/1040) > None (883/1040)
                # saga: l2 (894/1040) > None (891/1040) > elasticnet, l1_ratio=0.5 (888/1040) > l1 (866/1040)
                # lbfgs: l2 (894/1040) > None (855/1040)
            # Default Model with AllBothTestPreds: 894/1040
            # Test for AllTest13Preds with multi_class="multinomial":
            # solver and penalty:
                # newton-cg: l2 (44/56) > None (42/56)
                # sag: l2 (44/56) = None (44/56)
                # saga: l2 (44/56) = None (44/56) > elasticnet, l1_ratio=0.5 (42/56) > l1 (41/56)
                # lbfgs: l2 (44/56) = None (44/56)
            # LogisticRegression(multi_class="multinomial"): 44/56
            meta_model_LR = LogisticRegression(multi_class="multinomial")
            training_LR = meta_model_LR.fit(train_stack, train_labels)
            joblib.dump(training_LR, "MetaModel_LR_"+ self.json_file + "_Top" + str(self.top_x) + ".joblib")
            return training_LR

    def hard_voting(self, model_predictions):
        """
        model_predictions: list of tupels (model, top_x_predictions)
        """

        all_predicted_classes = {}

        for predictions in model_predictions:
            for predicted_classes in predictions[1]:
                if predicted_classes[1] > 0:  # only give a vote if probability > 0
                    if predicted_classes[0] in all_predicted_classes.keys():
                        x = all_predicted_classes.get(predicted_classes[0])
                        new = x + 1
                        all_predicted_classes.update({predicted_classes[0]: new})
                    else:
                        all_predicted_classes.update({predicted_classes[0]: 1})

        max_votes = 0
        for key, value in all_predicted_classes.items():
            if value > max_votes:
                max_votes = value
            else:
                continue

        most_voted = []
        for key, value in all_predicted_classes.items():
            if value == max_votes:
                most_voted.append((key, value))

        # Use simple soft voting when no unique maximum exists.
        if len(most_voted) > 1:
            mv = tuple(self.simple_soft_voting(model_predictions))
            most_voted = [mv]

        # print(most_voted)
        return most_voted

    def simple_soft_voting(self, model_predictions):
        """
        model_predictions: list of tupels (model, top_x_predictions)
        """
        all_predicted_classes = {}  # this dictionary saves the summed probability of the predicted classes

        for predictions in model_predictions:
            for predicted_classes in predictions[1]:
                if predicted_classes[0] in all_predicted_classes.keys():
                    x = all_predicted_classes.get(predicted_classes[0])
                    new = x + predicted_classes[1]
                    all_predicted_classes.update({predicted_classes[0]: new})
                else:
                    all_predicted_classes.update({predicted_classes[0]: predicted_classes[1]})

        max_prob = 0
        class_name = ''
        for key, value in all_predicted_classes.items():
            if value > max_prob:
                max_prob = value
                class_name = key
            else:
                continue

        # print([class_name, max_prob])
        return [class_name, max_prob]

    def weighted_soft_voting(self, model_predictions):
        """
        Same as simple_soft_voting, but weights the both models more.
        model_predictions: list of tupels (model, top_x_predictions)
        """

        all_predicted_classes = {}  # this dictionary saves the weighted sum of the probability of the predicted classes

        for predictions in model_predictions:
            for predicted_classes in predictions[1]:
                # Obv and Rev don't get a weight, but Both is weighted with 1.5.
                # predictions[0] saves the model name. If "Both" is part of the name set the weight.
                if predicted_classes[0] in all_predicted_classes.keys():
                    if "Both" in predictions[0]:
                        x = all_predicted_classes.get(predicted_classes[0])
                        new = x + 1.5*predicted_classes[1]
                        all_predicted_classes.update({predicted_classes[0]: new})
                    else:
                        x = all_predicted_classes.get(predicted_classes[0])
                        new = x + 1*predicted_classes[1]
                        all_predicted_classes.update({predicted_classes[0]: new})
                else:
                    if "Both" in predictions[0]:
                        all_predicted_classes.update({predicted_classes[0]: 1.5*predicted_classes[1]})
                    else:
                        all_predicted_classes.update({predicted_classes[0]: 1*predicted_classes[1]})

        max_prob = 0
        class_name = ''
        for key, value in all_predicted_classes.items():
            if value > max_prob:
                max_prob = value
                class_name = key
            else:
                continue

        # print([class_name, max_prob])
        return [class_name, max_prob]

    def weighted_obv_rev_voting(self, model_predictions):
        """
        Same as simple_soft_voting, but weights the obv and rev models more.
        model_predictions: list of tupels (model, top_x_predictions)
        """

        # Extracting common part of the keys
        common_key = {}
        for key, prediction in model_predictions:
            prefix = key.split('_')[0]  # Extracting common prefix
            if prefix not in common_key:
                common_key[prefix] = []
            common_key[prefix].append((key, prediction[0]))  # Storing the key and its prediction

        # Comparing the values and updating the values in model_predictions
        for prefix, values in common_key.items():
            obv_values = [prediction for key, prediction in values if 'Obv' in key]
            rev_values = [prediction for key, prediction in values if 'Rev' in key]
            if len(obv_values) > 0 and len(rev_values) > 0:
                if obv_values[0] == rev_values[0]:
                    # print(f"{prefix}_Obv and {prefix}_Rev have the same prediction:", obv_values[0])
                    # Update the values in model_predictions for prefix_Obv and prefix_Rev with 2
                    for i in range(len(model_predictions)):
                        if f"{prefix}_Obv" in model_predictions[i][0]:
                            model_predictions[i][1][1] = 1.5 * model_predictions[i][1][1]
                        elif f"{prefix}_Rev" in model_predictions[i][0]:
                            model_predictions[i][1][1] = 1.5 * model_predictions[i][1][1]
        # Summing up the prob for each class
        sum_dict = defaultdict(float) # needed:   from collections import defaultdict 
        for key, prediction in model_predictions:
            name = prediction[0]
            prob = prediction[1]
            sum_dict[name] += prob

        # Finding the first prediction with the highest sum
        max_prob = 0
        class_name = None
        for key, prediction in sum_dict.items():
            if prediction > max_prob:
                max_prob = prediction
                class_name = key

        # print([class_name, max_prob])
        return [class_name, max_prob]

    def weighted_pred_soft_voting(self, model_predictions):
        """
        Same as simple_soft_voting, but weights the predictions of each model.
        model_predictions: list of tupels (model, top_x_predictions)
        """

        all_predicted_classes = {}  # this dictionary saves the summed probability of the predicted classes

        for predictions in model_predictions:
            for predicted_classes in predictions[1]:
                if predicted_classes[0] in all_predicted_classes.keys():
                    x = all_predicted_classes.get(predicted_classes[0])
                    # It takes into account the top-x position. 
                    # E.g.: if top-3 is given, the highest prediction gets the weight 3, the second the weight 2 and the lowest the weight 1.
                    # If onlny top-1 is given, it is the same as simple_soft_voting.
                    new = x + (predicted_classes[1] * (predictions[1].index(predicted_classes) + 1))
                    all_predicted_classes.update({predicted_classes[0]: new})
                else:
                    all_predicted_classes.update({predicted_classes[0]: predicted_classes[1]})

        max_prob = 0
        class_name = ''
        for key, value in all_predicted_classes.items():
            if value > max_prob:
                max_prob = value
                class_name = key
            else:
                continue

        # print([class_name, max_prob])
        return [class_name, max_prob]

    def run_hard_voting(self):
        """
        Function to run the voting.
        Saves the amount of right and wrong predictions and prints it.
        If plot==True the voting is evaluated and a confusion matrix and csv file is generated.
        """
        prediction_count = {"images": 0, "hv_prediction": 0}
        types_images = {}
        right_types_hv_prediction = {}
        wrong_types_hv_prediction = {}
        for value in self.dict_types.values():
            types_images.update({value: 0})
            right_types_hv_prediction.update({value: 0})
            wrong_types_hv_prediction.update({value: 0})

        hv_predictions = []
        true_labels = []

        self.load_test_predictions_array()

        for i in range(0, len(self.test_predictions_array[0][1])):
            prediction_count.update({"images": prediction_count["images"] + 1})
            all_predictions = []
            for prediction in self.test_predictions_array:
                true_label = self.dict_types[list(prediction[1][i][1]).index(1)]
                all_predictions.append((prediction[0], self.top_x_predictions(prediction[1][i][0])))
            types_images.update({true_label: types_images[true_label] + 1})
            hv_prediction = self.hard_voting(all_predictions)

            hv_predictions.append(hv_prediction[0][0])
            true_labels.append(true_label)

            if true_label in hv_prediction[0]:
                prediction_count.update({"hv_prediction": prediction_count["hv_prediction"] + 1})
                right_types_hv_prediction.update({true_label: right_types_hv_prediction[true_label] + 1})
            elif true_label not in hv_prediction[0]:
                wrong_types_hv_prediction.update({hv_prediction[0][0]: wrong_types_hv_prediction[hv_prediction[0][0]] + 1})

        print(prediction_count)
        print("True labels of the prediction:")
        print(true_labels)
        print("Predictions of the hard voting:")
        print(hv_predictions)
        print("Right predictions of the hard voting:")
        print(right_types_hv_prediction)
        print("Wrong predictions of the hard voting:")
        print(wrong_types_hv_prediction)
        self.base_learner_predictions()

        if self.plot == True:
            self.evaluate_ensemble(true_labels, hv_predictions, types_images, "hv")

        return hv_predictions

    def run_simple_soft_voting(self):
        """
        Function to run the voting.
        Saves the amount of right and wrong predictions and prints it.
        If plot==True the voting is evaluated and a confusion matrix and csv file is generated.
        """
        prediction_count = {"images": 0, "ssv_prediction": 0}
        types_images = {}
        right_types_ssv_prediction = {}
        wrong_types_ssv_prediction = {}
        for value in self.dict_types.values():
            types_images.update({value: 0})
            right_types_ssv_prediction.update({value: 0})
            wrong_types_ssv_prediction.update({value: 0})

        ssv_predictions = []
        true_labels = []

        self.load_test_predictions_array()

        for i in range(0, len(self.test_predictions_array[0][1])):
            prediction_count.update({"images": prediction_count["images"] + 1})
            all_predictions = []
            for prediction in self.test_predictions_array:
                true_label = self.dict_types[list(prediction[1][i][1]).index(1)]
                all_predictions.append((prediction[0], self.top_x_predictions(prediction[1][i][0])))
            types_images.update({true_label: types_images[true_label] + 1})
            ssv_prediction = self.simple_soft_voting(all_predictions)

            ssv_predictions.append(ssv_prediction[0])
            true_labels.append(true_label)

            if true_label in ssv_prediction[0]:
                prediction_count.update({"ssv_prediction": prediction_count["ssv_prediction"] + 1})
                right_types_ssv_prediction.update({true_label: right_types_ssv_prediction[true_label] + 1})
            elif true_label not in ssv_prediction[0]:
                wrong_types_ssv_prediction.update({ssv_prediction[0]: wrong_types_ssv_prediction[ssv_prediction[0]] + 1})

        print(prediction_count)
        print("True labels of the prediction:")
        print(true_labels)
        print("Predictions of the simple soft voting:")
        print(ssv_predictions)
        print("Right predictions of the simple soft voting:")
        print(right_types_ssv_prediction)
        print("Wrong predictions of the simple soft voting:")
        print(wrong_types_ssv_prediction)
        self.base_learner_predictions()

        if self.plot == True:
            self.evaluate_ensemble(true_labels, ssv_predictions, types_images, "ssv")

        return ssv_predictions

    def run_weighted_soft_voting(self, plot=True):
        """
        Function to run the voting.
        Saves the amount of right and wrong predictions and prints it.
        If plot==True the voting is evaluated and a confusion matrix and csv file is generated.
        """
        prediction_count = {"images": 0, "wsv_prediction": 0}
        types_images = {}
        right_types_wsv_prediction = {}
        wrong_types_wsv_prediction = {}
        for value in self.dict_types.values():
            types_images.update({value: 0})
            right_types_wsv_prediction.update({value: 0})
            wrong_types_wsv_prediction.update({value: 0})

        wsv_predictions = []
        true_labels = []

        self.load_test_predictions_array()

        for i in range(0, len(self.test_predictions_array[0][1])):
            prediction_count.update({"images": prediction_count["images"] + 1})
            all_predictions = []
            for prediction in self.test_predictions_array:
                true_label = self.dict_types[list(prediction[1][i][1]).index(1)]
                all_predictions.append((prediction[0], self.top_x_predictions(prediction[1][i][0])))
            types_images.update({true_label: types_images[true_label] + 1})
            wsv_prediction = self.weighted_soft_voting(all_predictions)

            wsv_predictions.append(wsv_prediction[0])
            true_labels.append(true_label)

            if true_label in wsv_prediction[0]:
                prediction_count.update({"wsv_prediction": prediction_count["wsv_prediction"] + 1})
                right_types_wsv_prediction.update({true_label: right_types_wsv_prediction[true_label] + 1})
            elif true_label not in wsv_prediction[0]:
                wrong_types_wsv_prediction.update({wsv_prediction[0]: wrong_types_wsv_prediction[wsv_prediction[0]] + 1})

        print(prediction_count)
        print("True labels of the prediction:")
        print(true_labels)
        print("Predictions of the weighted soft voting:")
        print(wsv_predictions)
        print("Right predictions of the weighted soft voting:")
        print(right_types_wsv_prediction)
        print("Wrong predictions of the weighted soft voting:")
        print(wrong_types_wsv_prediction)
        self.base_learner_predictions()

        if self.plot == True:
            self.evaluate_ensemble(true_labels, wsv_predictions, types_images, "wsv")

        return wsv_predictions

    def run_weighted_obv_rev_voting(self, plot=True):
        """
        Function to run the voting.
        Saves the amount of right and wrong predictions and prints it.
        If plot==True the voting is evaluated and a confusion matrix and csv file is generated.
        """
        prediction_count = {"images": 0, "worv_prediction": 0}
        types_images = {}
        right_types_worv_prediction = {}
        wrong_types_worv_prediction = {}
        for value in self.dict_types.values():
            types_images.update({value: 0})
            right_types_worv_prediction.update({value: 0})
            wrong_types_worv_prediction.update({value: 0})

        worv_predictions = []
        true_labels = []

        self.load_test_predictions_array() # added num_of_elements in the function
        for i in range(self.num_of_test_elements): # added num_of_elements
            prediction_count.update({"images": prediction_count["images"] + 1})
            all_predictions = []
            for prediction in self.test_predictions_array:
                true_label = self.dict_types[list(prediction[1][i][1]).index(1)]
                all_predictions.append((prediction[0], self.top_1_predictions(prediction[1][i][0])))
            types_images.update({true_label: types_images[true_label] + 1})
            # print(all_predictions)
            # return
            worv_prediction = self.weighted_obv_rev_voting(all_predictions)   
            
            worv_predictions.append(worv_prediction[0])
            true_labels.append(true_label)

            # print(worv_prediction)

            if true_label in worv_prediction[0]:
                prediction_count.update({"worv_prediction": prediction_count["worv_prediction"] + 1})
                right_types_worv_prediction.update({true_label: right_types_worv_prediction[true_label] + 1})
            elif true_label not in worv_prediction[0]:
                wrong_types_worv_prediction.update({worv_prediction[0]: wrong_types_worv_prediction[worv_prediction[0]] + 1})

        print(prediction_count)
        print("True labels of the prediction:")
        print(true_labels)
        print("Predictions of the weighted obv rev voting:")
        print(worv_predictions)
        print("Right predictions of the weighted obv rev voting:")
        print(right_types_worv_prediction)
        print("Wrong predictions of the weighted obv rev voting:")
        print(wrong_types_worv_prediction)
        self.base_learner_predictions()

        if self.plot == True:
            self.evaluate_ensemble(true_labels, worv_predictions, types_images, "worv")

        return worv_predictions

    def run_weighted_pred_soft_voting(self, plot=True):
        """
        Function to run the voting.
        Saves the amount of right and wrong predictions and prints it.
        If plot==True the voting is evaluated and a confusion matrix and csv file is generated.
        """
        prediction_count = {"images": 0, "wpsv_prediction": 0}
        types_images = {}
        right_types_wpsv_prediction = {}
        wrong_types_wpsv_prediction = {}
        for value in self.dict_types.values():
            types_images.update({value: 0})
            right_types_wpsv_prediction.update({value: 0})
            wrong_types_wpsv_prediction.update({value: 0})

        wpsv_predictions = []
        true_labels = []

        self.load_test_predictions_array()

        for i in range(0, len(self.test_predictions_array[0][1])):
            prediction_count.update({"images": prediction_count["images"] + 1})
            all_predictions = []
            for prediction in self.test_predictions_array:
                true_label = self.dict_types[list(prediction[1][i][1]).index(1)]
                all_predictions.append((prediction[0], self.top_x_predictions(prediction[1][i][0])))
            types_images.update({true_label: types_images[true_label] + 1})
            wpsv_prediction = self.weighted_pred_soft_voting(all_predictions)

            wpsv_predictions.append(wpsv_prediction[0])
            true_labels.append(true_label)

            if true_label in wpsv_prediction[0]:
                prediction_count.update({"wpsv_prediction": prediction_count["wpsv_prediction"] + 1})
                right_types_wpsv_prediction.update({true_label: right_types_wpsv_prediction[true_label] + 1})
            elif true_label not in wpsv_prediction[0]:
                wrong_types_wpsv_prediction.update({wpsv_prediction[0]: wrong_types_wpsv_prediction[wpsv_prediction[0]] + 1})

        print(prediction_count)
        print("True labels of the prediction:")
        print(true_labels)
        print("Predictions of the weighted predictions soft voting:")
        print(wpsv_predictions)
        print("Right predictions of the weighted predictions soft voting:")
        print(right_types_wpsv_prediction)
        print("Wrong predictions of the weighted predictions soft voting:")
        print(wrong_types_wpsv_prediction)
        self.base_learner_predictions()

        if self.plot == True:
            self.evaluate_ensemble(true_labels, wpsv_predictions, types_images, "wpsv")

        return wpsv_predictions

    def run_stacking_RF(self):
        """
        Function to run the stacking.
        Generates first the meta data with the validationset and build the meta model.
        Meta model is evaluated with the testdata.
        Saves the amount of right and wrong predictions and prints it.
        If plot==True the voting is evaluated and a confusion matrix and csv file is generated.
        """
        # Build first the meta model with the validation set
        self.load_val_predictions_array()
        train_stack = self.stack_val_predictions()
        labels = self.build_label_array(self.val_predictions_array)
        train_labels = np.array(labels[0])
        meta_model_type = "RF"
        # print("Train labels: ", len(train_labels), type(train_labels), "\n", train_labels)
        # print("Predictions (train):", train_stack.shape, type(train_stack), "\n", train_stack)
        meta_model = self.train_meta_model(train_stack, train_labels, meta_model_type)
        print("Training of the meta model finished!")

        # Make predictions on the testset
        prediction_count = {"images": 0, "stackingRF_prediction": 0}
        types_images = {}
        right_types_stackingRF_prediction = {}
        for value in self.dict_types.values():
            types_images.update({value: 0})
            right_types_stackingRF_prediction.update({value: 0})

        self.load_test_predictions_array()
        stacked_pred = self.stack_test_predictions()
        stackingRF_predictions =  meta_model.predict(stacked_pred)
        stackingRF_labels = self.stack_predictons_labels(meta_model, stacked_pred, stackingRF_predictions)

        true_labels = []
        predicted_labels = list(stackingRF_labels)

        if len(stackingRF_predictions) == len(self.test_predictions_array[0][1]):
            for i in range(0, len(self.test_predictions_array[0][1])):
                prediction_count.update({"images": prediction_count["images"] + 1})
                for prediction in self.test_predictions_array:
                    true_label = self.dict_types[list(prediction[1][i][1]).index(1)]
                types_images.update({true_label: types_images[true_label] + 1})

                true_labels.append(true_label)

                if true_label == predicted_labels[i]:
                    prediction_count.update({"stackingRF_prediction": prediction_count["stackingRF_prediction"] + 1})
                    right_types_stackingRF_prediction.update({true_label: right_types_stackingRF_prediction[true_label] + 1})

        print(prediction_count)
        print(right_types_stackingRF_prediction)
        self.base_learner_predictions()

        if self.plot == True:
            self.evaluate_ensemble(true_labels, predicted_labels, types_images, "stackingRF_prediction")

        return stackingRF_predictions

    def run_stacking_LR(self):
        """
        Function to run the stacking.
        Generates first the meta data with the validationset and build the meta model.
        Meta model is evaluated with the testdata.
        Saves the amount of right and wrong predictions and prints it.
        If plot==True the voting is evaluated and a confusion matrix and csv file is generated.
        """
        # Build first the meta model with the validation set
        self.load_val_predictions_array()
        train_stack = self.stack_val_predictions()
        labels = self.build_label_array(self.val_predictions_array)
        train_labels = np.array(labels[1])
        meta_model_type = "LR"
        # print("Train labels: ", len(train_labels), type(train_labels), "\n", train_labels)
        # print("Predictions (train):", train_stack.shape, type(train_stack), "\n", train_stack)
        meta_model = self.train_meta_model(train_stack, train_labels, meta_model_type)
        print("Training of the meta model finished!")

        # Make predictions on the testset
        prediction_count = {"images": 0, "stackingLR_prediction": 0}
        types_images = {}
        right_types_stackingLR_prediction = {}
        for value in self.dict_types.values():
            types_images.update({value: 0})
            right_types_stackingLR_prediction.update({value: 0})

        self.load_test_predictions_array()
        stacked_pred = self.stack_test_predictions()
        stackingLR_predictions =  meta_model.predict(stacked_pred)

        true_labels = []
        predicted_labels = list(stackingLR_predictions)

        if len(stackingLR_predictions) == len(self.test_predictions_array[0][1]):
            for i in range(0, len(self.test_predictions_array[0][1])):
                prediction_count.update({"images": prediction_count["images"] + 1})
                for prediction in self.test_predictions_array:
                    true_label = self.dict_types[list(prediction[1][i][1]).index(1)]
                types_images.update({true_label: types_images[true_label] + 1})

                true_labels.append(true_label)

                if true_label == predicted_labels[i]:
                    prediction_count.update({"stackingLR_prediction": prediction_count["stackingLR_prediction"] + 1})
                    right_types_stackingLR_prediction.update({true_label: right_types_stackingLR_prediction[true_label] + 1})

        print(prediction_count)
        print(right_types_stackingLR_prediction)
        self.base_learner_predictions()

        if self.plot == True:
            self.evaluate_ensemble(true_labels, predicted_labels, types_images, "stackingLR_prediction")

        return stackingLR_predictions

    def base_learner_predictions(self):
        prediction_count = {"images": 0}
        types_images = {}
        for value in self.dict_types.values():
            types_images.update({value: 0})

        cnn_dict = {}
        for cnn in self.networks:
            cnn_dict[f'right_types_{cnn}'] = types_images.copy()

        true_labels = []

        # Predictions already loaded.
        for entry in self.test_predictions_array:
            prediction_count.update({entry[0]: 0})

        for i in range(0, len(self.test_predictions_array[0][1])):
            prediction_count.update({"images": prediction_count["images"] + 1})
            all_predictions = []
            for prediction in self.test_predictions_array:
                true_label = self.dict_types[list(prediction[1][i][1]).index(1)]
                all_predictions.append((prediction[0], self.top_x_predictions(prediction[1][i][0])))
            types_images.update({true_label: types_images[true_label] + 1})

            true_labels.append(true_label)

            for element in all_predictions:
                if element[1][-1][0] == true_label:
                    prediction_count.update({element[0]: prediction_count[element[0]] + 1})

                for cnn in self.networks:
                    if cnn in element[0] and element[1][-1][0] == true_label:
                        cnn_dict[f'right_types_{cnn}'].update(
                            {true_label: cnn_dict[f'right_types_{cnn}'][true_label] + 1})

        print(prediction_count)

    def evaluate_ensemble(self, true_labels, ensemble_predictions, types_images, save_name):
        """
        Evaluates the ensembles.
        Results are printed and saved in confusion matrices and csv files.
        """
        plt.rcParams['axes.grid'] = False

        only_used_labels= []
        for label in true_labels:
            if int(label) not in only_used_labels:
                only_used_labels.append(int(label))

        only_used_labels.sort()

        only_used_labels = list(map(str, only_used_labels))

        # print(only_used_labels)
        for label in list(types_images.keys()):
            if label not in only_used_labels and (label in true_labels or label in ensemble_predictions):
                only_used_labels.append(label)

        print("only used labels")
        print(only_used_labels)

        cm = confusion_matrix(true_labels, ensemble_predictions, labels=only_used_labels, normalize='true')
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=only_used_labels)
        fig, ax = plt.subplots(figsize=(30, 30))
        cm_display.plot(ax=ax, xticks_rotation=90, include_values=False)
        plt.savefig("confusion_matrix_" + save_name + "_" + self.json_file + "_Top" + str(self.top_x) + ".jpg")

        self.print_score(true_labels, ensemble_predictions, only_used_labels, save_name)
        plt.show()

    def print_score(self, true_labels, ensemble_predictions, only_used_labels, save_name):
        """
        Prints the results of the evaluation.
        """
        clf_report = pd.DataFrame(
            classification_report(true_labels, ensemble_predictions, labels=only_used_labels, target_names=only_used_labels, output_dict=True, zero_division=0.0))
        clf_report.to_csv("classification_report_" + save_name + "_" + self.json_file + "_Top" + str(self.top_x) + ".csv")
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true_labels, ensemble_predictions) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Error Rate Score: {(1-accuracy_score(true_labels, ensemble_predictions)) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Precision Score: {precision_score(true_labels, ensemble_predictions, average='macro') * 100:.2f}%")
        print("_______________________________________________")
        print(f"Recal Score: {recall_score(true_labels, ensemble_predictions, average='macro') * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")


def main(json_file, dict_types, ensemble_type, top_x, eval):
    """
    Given the user input different ensembles and combination functions can be tested.
    """
    if ensemble_type == "hard_voting":
        hard_voting = Ensemble(json_file, dict_types, top_x, eval)
        hard_voting.run_hard_voting()
    elif ensemble_type == "simple_soft_voting":
        simple_soft_voting = Ensemble(json_file, dict_types, top_x, eval)
        simple_soft_voting.run_simple_soft_voting()
    elif ensemble_type == "weighted_soft_voting":
        weighted_soft_voting = Ensemble(json_file, dict_types, top_x, eval)
        weighted_soft_voting.run_weighted_soft_voting()
    elif ensemble_type == "weighted_obv_rev_voting":
        weighted_obv_rev_voting = Ensemble(json_file, dict_types, "1", eval)
        weighted_obv_rev_voting.run_weighted_obv_rev_voting()
    elif ensemble_type == "weighted_pred_soft_voting":
        weighted_pred_soft_voting = Ensemble(json_file, dict_types, top_x, eval)
        weighted_pred_soft_voting.run_weighted_pred_soft_voting()
    elif ensemble_type == "stacking_with_RF":
        stacking = Ensemble(json_file, dict_types, "1", eval)
        stacking.run_stacking_RF()
    elif ensemble_type == "stacking_with_LR":
        stacking = Ensemble(json_file, dict_types, "1", eval)
        stacking.run_stacking_LR()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for the ensemble.")

    parser.add_argument("json_file", help="File with all networks and their evaluation and test predictions path.")
    parser.add_argument("dict_types", help="Path to the pickle with the types.")
    parser.add_argument("ensemble_type", help="Select the ensemble method: hard_voting, simple_soft_voting, weighted_soft_voting, weighted_pred_soft_voting, weighted_obv_rev_voting, stacking_with_RF or stacking_with_LR.")
    parser.add_argument("top_x", help="Number of the wanted top predictions. Only for hard_voting, simple_soft_voting, weighted_soft_voting and weighted_pred_soft_voting relevant.")
    parser.add_argument("--eval", action="store_true", help="True for evaluating, plotting and saving the results.")
    args = parser.parse_args()
    main(args.json_file, args.dict_types, args.ensemble_type, args.top_x, args.eval)