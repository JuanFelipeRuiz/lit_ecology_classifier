import os
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch



def plot_reduced_classes(model, priority_classes):
    """
    Plots the confusion matrix for reduced classes.

    Args:
        model (LightningModule): The trained model.
        priority_classes (list): List of priority classes.

    Saves:
        reduced_confusion_matrix.png: A confusion matrix of the reduced classes.
        reduced_confusion_matrix_norm.png: A normalized confusion matrix of the reduced classes.
    """
    reduced_class_map = {v: k + 1 for k, v in enumerate(priority_classes)}
    reduced_class_map["rest"] = 0
    inv_reduced_class_map = {v: k for k, v in reduced_class_map.items()}
    reduced_preds = []
    reduced_labels = []
    preds = torch.cat(model.test_step_predictions)
    true_labels = torch.cat(model.test_step_targets)
    for pred, true in zip(preds, true_labels):
        name = model.inverted_class_map[pred.item()]
        name2 = model.inverted_class_map[true.item()]
        reduced_preds.append(reduced_class_map[name] if name in reduced_class_map else 0)
        reduced_labels.append(reduced_class_map[name2] if name2 in reduced_class_map else 0)
    all_preds = torch.tensor(reduced_preds)
    all_labels = torch.tensor(reduced_labels)
    fig, fig2 = plot_confusion_matrix(all_labels, all_preds, inv_reduced_class_map)
    fig.savefig(f"{model.outpath}/reduced_confusion_matrix.png")
    fig2.savefig(f"{model.outpath}/reduced_confusion_matrix_norm.png")


def plot_loss_acc(logger):
    """
    Plots the training and validation loss and accuracy from the logger's metrics file.

    Args:
        logger (Logger): The logger object containing the save directory, name, and version.

    Saves:
        loss_accuracy.png: A plot of the training and validation loss and accuracy over steps.
    """
    # Read the CSV file
    metric_path = os.path.join(logger.save_dir, logger.name, f"version_{logger.version}", "metrics.csv")
    metrics_df = pd.read_csv(metric_path)

    # Plot the training loss
    step = metrics_df["step"]
    train_loss = metrics_df["train_loss"]
    val_loss = metrics_df["val_loss"]
    train_acc = metrics_df["train_acc"]
    val_acc = metrics_df["val_acc"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # usage of the mask to remove NaN values (numpy.nan != numpy.nan)
    ax[0].plot(step[train_loss == train_loss], train_loss[train_loss == train_loss], label="Training Loss", color="skyblue")
    ax[0].plot(step[val_loss == val_loss], val_loss[val_loss == val_loss], label="Validation Loss", color="crimson")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss vs Steps")
    ax[0].legend()

    ax[1].plot(step[train_loss == train_loss], train_acc[train_loss == train_loss], label="Training Accuracy", color="skyblue")
    ax[1].plot(step[val_loss == val_loss], val_acc[val_loss == val_loss], label="Validation Accuracy", color="crimson")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy vs Steps")
    ax[1].legend()
    fig.tight_layout()
    plt.savefig(f"{logger.save_dir}/{logger.name}/version_{logger.version}/loss_accuracy.png")


def plot_score_distributions(all_scores, all_preds, class_names, true_label):
    """
    Plot the distribution of prediction scores for each class in separate plots.

    Args:
        all_scores (torch.Tensor): Confidence scores of the predictions.
        all_preds (torch.Tensor): Predicted class indices.
        class_names (list): List of class names.

    Returns:
        list: A list of figures, each representing the score distribution for a class.
    """
    # Convert scores and predictions to CPU if not already
    all_scores = all_scores.cpu().numpy()
    all_preds = all_preds.cpu().numpy()
    true_label = true_label.cpu().numpy()
    # List to hold the figures
    fig, ax = plt.subplots(len(class_names) // 4 + 1, 4, figsize=(20, len(class_names) // 4 * 5 + 1))
    ax = ax.flatten()

    # Creating a histogram for each class
    for i, class_name in enumerate(class_names):
        # Filter scores for predictions matching the current class
        sig_scores = all_scores[(true_label == i)][:, i]
        bkg_scores = all_scores[(true_label != i)][:, i]
        # Create a figure for the current class
        ax[i].hist(bkg_scores, bins=np.linspace(0, 1, 30), color="skyblue", edgecolor="black")
        ax[i].set_ylabel("Rest Counts", color="skyblue")
        ax[i].set_yscale("log")
        y_axis = ax[i].twinx()
        y_axis.hist(sig_scores, bins=np.linspace(0, 1, 30), color="crimson", histtype="step", edgecolor="crimson")
        ax[i].set_title(f"{class_name}")
        ax[i].set_xlabel("Predicted Probability")
        y_axis.set_ylabel("Signal Counts", color="crimson")
        y_axis.set_yscale("log")
    fig.tight_layout()
    return fig


def cvd_colormap():
    """
    A color map accessible for people with color vision deficiency (CVD).
    """
    stops = [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000]
    red = [0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764]
    green = [0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832]
    blue = [0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539]

    # Create a dictionary with color information
    cdict = {
        'red': [(stops[i], red[i], red[i]) for i in range(len(stops))],
        'green': [(stops[i], green[i], green[i]) for i in range(len(stops))],
        'blue': [(stops[i], blue[i], blue[i]) for i in range(len(stops))]
    }

    # Create the colormap
    return LinearSegmentedColormap('CustomMap', segmentdata=cdict, N=255)


def plot_confusion_matrix(all_labels, all_preds, class_names):
    """
    Plots a confusion matrix and a normalized confusion matrix for the given predictions.

    Args:
        all_labels: can be a numpy array or a torch tensor of class labels
        all_preds: can be a numpy array or a torch tensor of predicted class labels
        class_names (list): List of class names.

    Returns:
        A tuple containing the plots: (confusion matrix, normalized confusion matrix)
    """
    # TODO: Check if this is necessary
    if not isinstance(all_labels, np.ndarray):
        all_labels = all_labels.cpu().numpy()
    if not isinstance(all_preds, np.ndarray):
        all_preds = all_preds.cpu().numpy()

    class_indices = np.arange(len(class_names))
    confusion_matrix = sklearn.metrics.confusion_matrix(all_labels, all_preds, labels=class_indices)
    confusion_matrix_norm = sklearn.metrics.confusion_matrix(all_labels, all_preds, normalize="pred", labels=class_indices)
    num_classes = confusion_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(20, 20))
    fig2, ax2 = plt.subplots(figsize=(20, 20))


    if len(class_names) != num_classes:
        print(f"Warning: Number of class names ({len(class_names)}) does not match the number of classes ({num_classes}) in confusion matrix.")
        class_names = class_names[:num_classes]
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=class_names)
    cm_display_norm = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix_norm, display_labels=class_names)
    cmap = cvd_colormap()
    cm_display.plot(cmap=cmap, ax=ax, xticks_rotation=90)
    cm_display_norm.plot(cmap=cmap, ax=ax2, xticks_rotation=90)

    fig.tight_layout()
    fig2.tight_layout()
    return fig, fig2



def compute_roc_auc_binary(all_labels, all_scores, debug=False):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    import numpy as np
    import os

    # Ensure the output path exists

    # Convert tensors to NumPy arrays
    all_labels_np = all_labels.cpu().numpy()
    all_scores_np = all_scores.cpu().numpy()

    # Create binary labels: 0 for class 0 (negative), 1 for all other classes (positive)
    binary_labels = np.where(all_labels_np == 0, 0, 1)

    # Compute scores for the positive class (all classes except 0)
    # Sum the scores of all positive classes to get a single score per sample
    positive_scores = all_scores_np[:, 1:].sum(axis=1)

    # Use the positive class scores as the prediction scores
    y_true = binary_labels
    y_scores = positive_scores

    # Compute AUC for the binary classification
    if len(np.unique(y_true)) > 1:
        auc_score = roc_auc_score(y_true, y_scores)
    else:
        # If only one class present in y_true, AUC is not defined
        auc_score = float('nan')
    if debug:
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        os.makedirs('debug', exist_ok=True)

        # Plot ROC curve
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color='blue',
            lw=2,
            label='ROC curve (AUC = %0.2f)' % auc_score
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Binary Classification (Class 0 vs. Others)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join('debug', 'roc_curve_binary.png')
        )
        plt.close()

        # Plot score distribution for the binary classification
        plt.figure()
        plt.hist(
            y_scores[y_true == 1],
            bins=50,
            alpha=0.5,
            label='Positive (Classes 1 and above)',
            color='blue',
        )
        plt.hist(
            y_scores[y_true == 0],
            bins=50,
            alpha=0.5,
            label='Negative (Class 0)',
            color='orange',
        )
        plt.title('Score Distribution for Binary Classification\nAUC: {:.2f}'.format(auc_score))
        plt.xlabel('Aggregated Positive Class Score')
        plt.ylabel('Frequency')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join('debug', 'score_distribution_binary.png')
        )
        plt.close()
    return auc_score

def plot_roc_curve(fpr, tpr,auc_score, class_label, path = ''):
    plt.figure()
    plt.plot(  fpr,    tpr,
                    color='blue',
                    lw=2,
                    label='ROC curve (AUC = %0.2f)' % auc_score
            )
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Class {}'.format(class_label))
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    if path != '':
        path = Path(path) / "roc_curve_class_{}.png".format(class_label)
        plt.savefig(path)
    return plt

def plot_score_distribution_single_class(y_true, y_scores, auc_score, class_label,path = ''):

    plt.figure()
    plt.hist(
                y_scores[y_true == 1],
                bins=50,
                alpha=0.5,
                label='Positive (Class {})'.format(class_label),
                color='blue',
            )
    plt.hist(
                y_scores[y_true == 0],
                bins=50,
                alpha=0.5,
                label='Negative (Other Classes)',
                color='orange',
            )
    plt.title('Score Distribution for Class {}, AUC: {:.2f}'.format(class_label, auc_score))
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()

    if path != '':
        path = Path(path) / "score_distribution_class_{}.png".format(class_label)
        plt.savefig(path)
    return plt

def barplot_predictions(pred_labels, inverted_class_map, path = ''):
    """
    Plot the distribution of prediction scores for each class in a histogram.
    """
    classes = [inverted_class_map[k] for k in sorted(inverted_class_map.keys())]

    classes = [ class_name.replace("_", " ") for class_name in classes]
    
    # Count frequency of each class in pred_labels.
    counts = {cls: 0 for cls in classes}
    for label in pred_labels:
        counts[label] += 1
    

    frequencies = [counts[cls] for cls in classes]
    
    # Plot the bar chart.
    plt.figure(figsize=(10, 6))
    plt.bar(classes, frequencies)
    # add the count of each class on top of the bar
    for i, freq in enumerate(frequencies):
        if freq < 100:
            # vertical alignment for small frequencies
            plt.text(i, freq, freq, ha='center', va='bottom')

        else:
            # horizontal alignment for large frequencies
            plt.text(i, freq, freq, ha='center', va='bottom', rotation=90, fontsize=8)

    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.title("Predictions Histogram")
    plt.xticks(rotation=90, fontsize = 8)
    plt.tight_layout()
    plt.show()
    
    if path:
        path = Path(path) / "predictions_histogram.png"
        plt.savefig(path)
        plt.close()
    