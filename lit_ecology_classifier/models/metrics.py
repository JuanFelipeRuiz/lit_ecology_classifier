"""
Metrics
=======

This module provides metrics calculation and evaluation tools
for machine learning model assessment, particularly focused on classification tasks.

Features
--------
- Calculation of standard classification metrics (accuracy, precision, recall, F1)
- ROC-AUC computation for multi-class problems
- Population count analysis and bias metrics
- Special handling for plankton classification with junk categories

Main Components
-------------
* extra_metrics : Population-based metrics calculation
* compute_metrics : Basic classification performance metrics
* compute_roc_auc : ROC curve and AUC score computation
"""

from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from lit_ecology_classifier.helpers.modelling_plots import plot_roc_curve, plot_score_distribution_single_class


def extra_metrics(GT_label, Pred_label, Pred_prob, ID_result):
    """
    Calculate Bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk
    and return the dataframe of the population count. 

    Args:
        GT_label: Ground truth/ True label
        Pred_label: Predicted label
        Pred_prob: Predicted probability
        ID_result: ID of the result
    """

    list_class = list(set(np.unique(GT_label)).union(set(np.unique(Pred_label))))
    list_class.sort()
    df_count_Pred_GT = pd.DataFrame(index=list_class, columns=['Predict', 'Ground_truth'])

    GT_label_ID = ID_result[2].tolist()
    Pred_label_ID = ID_result[3].tolist()
    Pred_prob_ID = ID_result[4]

    list_class_ID = np.unique(GT_label_ID).tolist()
    list_class_ID.sort()
    df_prob = pd.DataFrame(index=list_class_ID, columns=['prob'])
    for i in range(len(list_class_ID)):
        df_prob.iloc[i] = np.sum(Pred_prob[:, i])

    df_prob_ID_all = pd.DataFrame(data=Pred_prob_ID, columns=list_class_ID)

    CC = []
    AC = []
    PCC = []
    PAC = []

    Pred_label = Pred_label.tolist()
    GT_label = GT_label.tolist()
    for iclass in list_class:
        df_count_Pred_GT.loc[iclass, 'Predict'] = Pred_label.count(iclass)
        df_count_Pred_GT.loc[iclass, 'Ground_truth'] = GT_label.count(iclass)

        class_CC = Pred_label.count(iclass)
        CC.append(class_CC)

        true_copy, pred_copy = GT_label_ID.copy(), Pred_label_ID.copy()
        for i in range(len(GT_label_ID)):
            if GT_label_ID[i] == iclass:
                true_copy[i] = 1
            else:
                true_copy[i] = 0
            if Pred_label_ID[i] == iclass:
                pred_copy[i] = 1
            else:
                pred_copy[i] = 0
        tn, fp, fn, tp = confusion_matrix(true_copy, pred_copy).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        class_AC = (class_CC - (fpr * len(Pred_label))) / (tpr - fpr)
        AC.append(class_AC)

        class_PCC = df_prob.loc[iclass, 'prob']
        PCC.append(class_PCC)

        df_prob_ID = pd.DataFrame()
        df_prob_ID['Pred_label'] = Pred_label_ID
        df_prob_ID['GT_label'] = GT_label_ID
        df_prob_ID['Pred_prob'] = df_prob_ID_all[iclass]
        tpr_prob = np.sum(df_prob_ID[(df_prob_ID['GT_label'] == iclass) & (df_prob_ID['Pred_label'] == iclass)]['Pred_prob']) / (tp + fn)
        fpr_prob = np.sum(df_prob_ID[(df_prob_ID['GT_label'] != iclass) & (df_prob_ID['Pred_label'] == iclass)]['Pred_prob']) / (tn + fp)
        class_PAC = (class_PCC - (fpr_prob * len(Pred_label))) / (tpr_prob - fpr_prob)
        PAC.append(class_PAC)

    df_percentage_Pred_GT = df_count_Pred_GT.div(df_count_Pred_GT.sum(axis=0), axis=1)
    df_count_Pred_GT['Bias'] = df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth']
    df_count_Pred_GT['CC'], df_count_Pred_GT['AC'], df_count_Pred_GT['PCC'], df_count_Pred_GT['PAC'] = CC, AC, PCC, PAC

    df_count_Pred_GT_rm_junk = df_count_Pred_GT.drop(['dirt', 'unknown', 'unknown_plankton'], errors='ignore')
    df_count_Pred_GT_rm_junk = df_count_Pred_GT_rm_junk.drop(df_count_Pred_GT_rm_junk[df_count_Pred_GT_rm_junk['Ground_truth'] == 0].index)

    df_count_Pred_GT_rm_0 = df_count_Pred_GT.drop(df_count_Pred_GT[df_count_Pred_GT['Ground_truth'] == 0].index)

    bias = np.sum(df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth']) / df_count_Pred_GT.shape[0]
    BC = np.sum(np.abs(df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth'])) / np.sum(np.abs(df_count_Pred_GT['Predict'] + df_count_Pred_GT['Ground_truth']))
    MAE = mean_absolute_error(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])
    MSE = mean_squared_error(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])
    RMSE = np.sqrt(MSE)
    R2 = r2_score(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])

    AE_rm_junk = np.sum(np.abs(df_count_Pred_GT_rm_junk['Predict'] - df_count_Pred_GT_rm_junk['Ground_truth']))
    NAE_rm_junk = np.sum(np.divide(np.abs(df_count_Pred_GT_rm_junk['Predict'] - df_count_Pred_GT_rm_junk['Ground_truth']), df_count_Pred_GT_rm_junk['Ground_truth']))
    NMAE = np.mean(np.divide(np.abs(df_count_Pred_GT_rm_0['Predict'] - df_count_Pred_GT_rm_0['Ground_truth']), df_count_Pred_GT_rm_0['Ground_truth']))

    return bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk, df_count_Pred_GT

def compute_metrics(all_y_labels, predicted_labels,inverted_class_map):
    """
    Compute the basic metrics for the model.
    """
    false_positives = torch.sum((all_y_labels == 0) & (predicted_labels != 0)) / torch.sum(
            all_y_labels == 0
        )

    all_labels_np = all_y_labels.cpu().numpy()
    predicted_labels_np = predicted_labels.cpu().numpy()

    balanced_acc = balanced_accuracy_score(all_labels_np, predicted_labels_np)
    macro_precision = precision_score(all_labels_np, predicted_labels_np, average='macro',labels=np.unique(all_labels_np))
    macro_recall = recall_score(all_labels_np, predicted_labels_np, average='macro',labels=np.unique(all_labels_np))
    macro_f1 = f1_score(all_labels_np, predicted_labels_np, average='macro',labels=np.unique(all_labels_np))
    accuracy_model = accuracy_score(all_labels_np, predicted_labels_np)
    clf_report = classification_report(all_labels_np, predicted_labels, labels=np.unique(all_labels_np), target_names=list(inverted_class_map.values()))       
    return balanced_acc, false_positives.item() , macro_precision, macro_recall, macro_f1, accuracy_model, all_labels_np, predicted_labels_np, clf_report



def compute_roc_auc(all_labels, all_scores, debug=False, path = ''): #debug logs some figures in a debug folder
    
    # Convert tensors to NumPy arrays
    if not isinstance(all_labels, np.ndarray):
        all_labels_np = all_labels.cpu().numpy()

    if not isinstance(all_scores, np.ndarray):
        all_scores_np = all_scores.cpu().numpy()

    # Get unique class labels
    class_labels = np.unique(all_labels_np)

    # Binarize the labels for multi-class ROC computation
    all_labels_binarized = label_binarize(all_labels_np, classes=class_labels)

    # Compute AUC for each class, plot score distributions, and plot ROC curves
    auc_list = []
    for i, class_label in enumerate(class_labels):
        try:
            
            y_true = all_labels_binarized[:, i]
            y_scores = all_scores_np[:, i]
        except:
            print("Error in class label", class_label)
            print("all_labels_binarized", all_labels_binarized)
            print("all_scores_np", all_scores_np)
            continue

        # Check if both classes are present
        if len(np.unique(y_true)) > 1:
            # Compute AUC for the class
            auc_score = roc_auc_score(y_true, y_scores)
            auc_list.append(auc_score)

            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            if debug:
                os.makedirs('debug', exist_ok=True)
                path = Path("debug")
                _ = plot_roc_curve(fpr, tpr, auc_score, class_label, path)
        else:
            # If only one class present in y_true, AUC and ROC are not defined
            auc_score = float('nan')
            auc_list.append(auc_score)
            # Skip plotting ROC curve
            pass

        # Plot score distribution for the class
        if debug:
            plot_score_distribution_single_class(y_true, y_scores, auc_score, class_label, path)

    # Compute macro-average AUC (ignoring NaN values)
    valid_auc_scores = [auc for auc in auc_list if not np.isnan(auc)]
    if valid_auc_scores:
        roc_auc_macro = np.mean(valid_auc_scores)
    else:
        roc_auc_macro = float('nan')

    return roc_auc_macro