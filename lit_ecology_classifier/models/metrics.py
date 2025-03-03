from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import torch
from lightning import LightningModule


def extra_metrics(GT_label, Pred_label, Pred_prob, ID_result):
    """
    Calculate Bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk
    and return the dataframe of the population count. 

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

def compute_metrics(all_labels, predicted_labels):
    """
    Compute the most important metrics for the test set.
    """
    false_positives = torch.sum((all_labels == 0) & (predicted_labels != 0)) / torch.sum(
            all_labels == 0
        )

    all_labels_np = all_labels.cpu().numpy()
    predicted_labels_np = predicted_labels.cpu().numpy()

    balanced_acc = balanced_accuracy_score(all_labels_np, predicted_labels_np)
    macro_precision = precision_score(all_labels_np, predicted_labels_np, average='macro')
    macro_recall = recall_score(all_labels_np, predicted_labels_np, average='macro')
    macro_f1 = f1_score(all_labels_np, predicted_labels_np, average='macro')
    accuracy_model = accuracy_score(all_labels_np, predicted_labels_np)         
    return balanced_acc, false_positives.item() , macro_precision, macro_recall, macro_f1, accuracy_model, all_labels_np, predicted_labels_np
    
def create_classification_report(all_labels, predicted_labels, accuracy,macro_f1, inverted_class_map):
    """
    Calculate the classification report and return the content of
    the file.

    Args:
        all_labels: Ground truth/ True label
        predicted_labels: Predicted label
        accuracy: Accuracy of the model
        macro_f1: Macro F1 score of the model
        inverted_class_map: Inverted class map

    Returns:
        Content to create a file with the overall classification report
    """
    if not isinstance(all_labels, np.ndarray):
        all_labels = all_labels.cpu().numpy()
    if not isinstance(predicted_labels, np.ndarray):
        predicted_labels = predicted_labels.cpu().numpy()


    clf_report = classification_report(all_labels, predicted_labels, labels=np.unique(all_labels), target_names=list(inverted_class_map.values()))
    file_content = '\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy, macro_f1, clf_report)

    return file_content