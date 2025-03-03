import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary, StochasticWeightAveraging
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from torch.autograd import Variable
import tarfile
import os
import importlib
import re
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def output_results(outpath, im_names, labels, scores,priority_classes=False,rest_classes=False, datapath = "", legacy=True):
    """
    Output the prediction results to a file.

    Args:
        outpath (str): Output directory path.
        im_names (list): List of image filenames.
        labels (list): List of predicted labels (label with the highest score).
        scores (list): List of predicted scores.(score with the highest probability)
        priority_classes (bool): If True, the priority classes are used.
        rest_classes (bool): If True, the rest classes are used.
        datapath (str): Path to the data to create a distinct output file.
        legacy (bool): If True, the output format is legacy.
    """

    labels = labels.tolist()
    data_folder_name = os.path.basename(datapath).split(".")[0] 
    base_filename = f"{outpath}/predictions_lit_ecology_classifier"+("_priority" if priority_classes else "")+("_rest" if rest_classes else "")+("_"+data_folder_name)
    file_path = f"{base_filename}.txt"
    if datapath.find(".tar") != -1:
        im_names = [img.name for img in im_names]

    if legacy:
        lines = [f"{img}------------------ {label}/{score}\n" for img, label,score in zip(im_names, labels,scores)]

    else:
        lines = [f"{img}, {label}, {score}"  for img, label,score in zip(im_names, labels,scores)]
    
    with open(file_path, "w+") as f:
            f.writelines(lines)

def test_output_results(outpath, im_names, true_labels, labels, score,  all_scores,  datapath=""):
    """
    Output the prediction results to a file.

    Args:
        outpath (str): Output directory path.
        im_names (list): List of image filenames.
        true_labels (list): List of true labels.
        labels (list): List of predicted labels (label with the highest score).
        score (list): List of predicted scores.
        all_scores (list): List of all predicted scores.
        datapath (str): Path to the data to create a distinct output file.
    """

    data_folder_name = os.path.basename(datapath).split(".")[0] 
    base_filename = f"{outpath}/test_values"+("_"+data_folder_name)
    file_path = f"{base_filename}.txt"
    if datapath.find(".tar") != -1:
        im_names = [img.name for img in im_names]

    lines = [f"{img}, {true_label}, {label}, {score}, {all_score}"  for img, true_label, label, score, all_score in zip(im_names, true_labels, labels, score, all_scores)]
    
    with open(file_path, "w+") as f:
            f.writelines(lines)


def gmean(input_x, dim):
    """
    Compute the geometric mean of the input tensor along the specified dimension.

    Args:
        input_x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the geometric mean.

    Returns:
        torch.Tensor: Geometric mean of the input tensor.
    """
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with cosine annealing and warmup.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        warmup (int): Number of warmup steps.
        max_iters (int): Total number of iterations.

    Methods:
        get_lr: Compute the learning rate at the current step.
        get_lr_factor: Compute the learning rate factor at the current step.
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch >= self.max_num_iters:
            lr_factor *= self.max_num_iters / epoch
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def define_priority_classes(priority_classes):
    class_map = {class_name: i + 1 for i, class_name in enumerate(priority_classes)}
    class_map["rest"] = 0
    return class_map

def define_rest_classes(priority_classes):
    class_map = {class_name: i for i, class_name in enumerate(priority_classes)}
    return class_map


def TTA_collate_fn(batch: dict , train: bool = False):
    """
    Collate function for test time augmentation (TTA).

    Args:
        batch (dict): Dict of tuples containing images and labels.

    Returns:
        batch_images: All rotations stacked row-wise
        batch_labels: Labels of the images
    """
    batch_images = {rot: [] for rot in ["0", "90", "180", "270"]}
    batch_labels = []
    if train:
        for rotated_images, label in batch:
            for rot in batch_images:
                batch_images[rot].append(rotated_images[rot])
            batch_labels.append(label)
        batch_images = {rot: torch.stack(batch_images[rot]) for rot in batch_images}
        batch_labels = torch.tensor(batch_labels)
        return batch_images, batch_labels

    else:
        for rotated_images in batch:
            for rot in batch_images:
                batch_images[rot].append(rotated_images[rot])
        batch_images = {rot: torch.stack(batch_images[rot]) for rot in batch_images}
        return batch_images



def setup_callbacks(priority_classes, ckpt_name):
    """
    Sets up callbacks for the training process.

    Args:
        priority_classes (list): List of priority classes to monitor for false positives.
        ckpt_name (str): The name of the checkpoint file.

    Returns:
        list: A list of configured callbacks including EarlyStopping, ModelCheckpoint, and ModelSummary.
    """
    callbacks = []
    ckpt_name = ckpt_name + "-{epoch:02d}-{val_acc:.4f}" if len(priority_classes) == 0 else ckpt_name + "-{epoch:02d}-{val_acc:.4f}-{val_false_positives:.4f}"
    monitor = "val_acc" if len(priority_classes) == 0 else "val_precision"
    mode = "max"
    callbacks.append(ModelCheckpoint(filename=ckpt_name, monitor=monitor, mode=mode, save_top_k=5))
    callbacks.append(ModelSummary())
    return callbacks


def setup_classmap(class_map = None, datapath="", priority_classes=[], rest_classes=[]):
    if class_map != {} and class_map is not None:
        logging.info(f"Using the provided class map.")
        return class_map

    if priority_classes != []:

        logging.info(f"Priority classes not None. Loading priority classes from {priority_classes}")

        logging.info(f"Priority classes set to: {priority_classes}")
        class_map = define_priority_classes(priority_classes)

    elif rest_classes != []:

        logging.info(f"rest classes not None. Defining clas map from {rest_classes}")
        class_map = define_rest_classes(rest_classes)

    # Load class map from JSON or extract it from the tar file if not present
    else:

        logging.info(f" Extracting class map from tar file.")
        class_map = _extract_class_map(datapath)

    return class_map


def _extract_class_map(tar_or_dir_path):
    """
    Extracts the class map from the contents of the tar file or directory and saves it to a JSON file.

    Arguments:
    tar_or_dir_path: str
        Path to the tar file or directory containing the images.

    Returns:
    dict
        A dictionary mapping class names to indices.
    """
    logging.info("Extracting class map.")
    class_map = {}

    if tarfile.is_tarfile(tar_or_dir_path):
        logging.info("Detected tar file.")
        with tarfile.open(tar_or_dir_path, "r") as tar:
            # Temporary set to track folders that contain images
            folders_with_images = set()

            # First pass: Identify folders containing images
            for member in tar.getmembers():
                if member.isdir():
                    continue  # Skip directories
                if member.isfile() and member.name.lower().endswith(("jpg", "jpeg", "png")):
                    class_name = os.path.basename(os.path.dirname(member.name))
                    folders_with_images.add(class_name)

            # Second pass: Build the class map only for folders with images
            for member in tar.getmembers():
                if member.isdir():
                    continue  # Skip directories
                class_name = os.path.basename(os.path.dirname(member.name))
                if class_name in folders_with_images:
                    if class_name not in class_map:
                        class_map[class_name] = []
                    class_map[class_name].append(member.name)

    elif os.path.isdir(tar_or_dir_path):
        logging.info("Detected directory.")
        for root, _, files in os.walk(tar_or_dir_path):
            for file in files:
                if file.lower().endswith(("jpg", "jpeg", "png")):
                    class_name = os.path.basename(root)
                    if class_name not in class_map:
                        class_map[class_name] = []
                    class_map[class_name].append(os.path.join(root, file))

    else:
        raise ValueError("Provided path is neither a valid tar file nor a directory.")

    # Create a sorted list of class names and map them to indices
    sorted_class_names = sorted(class_map.keys())
    logging.info(f"Found {len(sorted_class_names)} classes.")
    class_map = {class_name: idx for idx, class_name in enumerate(sorted_class_names)}

    return class_map

def check_existans_of_class(class_map: dict, class_list : list) -> None:
    """ Check if the classes inside the class list are present in the class map.

    Args:
        class_map: Contains the class labels and their corresponding values.
        class_list: List of classes to check if they are present in the class map.

    Raises:
        ValueError: If a class in the class list is not present in the class map.
    """

    for class_name in class_list:
        if class_name not in class_map:
            logger.error("Class %s not found in the class map.", class_name)
            raise ValueError(f"Class {class_name} not found in the class map.")

def class_transformers(classes_list : list) -> list:
    """Transforms the class list to lowercase and replace whitespaces with underscores.

    Args:
        classes_list: List of classes to transform.

    Returns:
        A list containing the class names in lowercase.
    """

    if classes_list == []:
        return []
    
    return [class_name.lower().replace(" ", "_") for class_name in classes_list]


def filter_class_mapping(
    class_map: dict, rest_classes: list[str] = [], priority_classes: list[str] = []
) -> dict:
    """Prepares the class map based on the provided rest and priority classes.

    To focus the training on specific classes, the class map is updated to set classes
    that are not in the priority classes to 0. The rest classes are to select, wich classes
    should be kept in the class map alongside the priority classes. Empty rest and priority
    classes will result in no filtering.

    Args:
        class_map: Contains the class labels and their corresponding values.
        priority_classes : List of classes that keep their original mapping value.
                           If atleast one class is defined, all other classes are set to 0.
                           If empty, no priority classes are set.

        rest_classes: Classes to keep alonside the priority classes in the class map.
                        If empty, no classes are removed.


    Returns:
        A dictionary containing the updated class labels isnide the rest classes and
        priority classes.

    Examples:
        Given:

        .. code-block:: python
            priority_classes = ["class1"]
            rest_classes = ["class2"]

            class_map = {
                "class1": 1,
                "class2": 2,
                "class3": 3
            }

        The resulting class map would be:

            {
                "class1": 1,
                "class3": 0
            }

    """
    logging.info(
        "Classes to keep based on defined priority classes, if emtpy no prio are set:%s \n \
            Classes to keep based on defined rest classes. If empty, no class are filtered out: %s",
        rest_classes,
        priority_classes,
    )
    # lower each element in the list
    priority_classes = class_transformers(priority_classes)
    rest_classes = class_transformers(rest_classes)

    check_existans_of_class(class_map, priority_classes)
    check_existans_of_class(class_map, rest_classes)

    class_map = {

        (key if key in priority_classes or priority_classes == [] else "rest"): 
        (value if key in priority_classes or priority_classes == [] else 0) 
        for key, value in class_map.items()

        # keep the class that are in the rest classes or the priority classes
        if (key in priority_classes) or (key in rest_classes) or (rest_classes == [])  
    }

    if "rest" not in class_map:
        # extract 1 from the class values to create a class with the value 0
        class_map = {key: value - 1 for key, value in class_map.items()}
        
    return class_map

def _extract_class_map(tar_or_dir_path):
    """
    Extracts the class map from the contents of the tar file or directory and saves it to a JSON file.

    Arguments:
    tar_or_dir_path: str
        Path to the tar file or directory containing the images.

    Returns:
    dict
        A dictionary mapping class names to indices.
    """
    logger.info("Extracting class map.")
    class_map = {}

    if os.path.isfile(tar_or_dir_path):

        if tarfile.is_tarfile(tar_or_dir_path):
            logger.info("Detected tar file.")
            with tarfile.open(tar_or_dir_path, "r") as tar:
                # Temporary set to track folders that contain images
                folders_with_images = set()

                # First pass: Identify folders containing images
                for member in tar.getmembers():
                    if member.isdir():
                        continue  # Skip directories
                    if member.isfile() and member.name.lower().endswith(("jpg", "jpeg", "png")):
                        class_name = os.path.basename(os.path.dirname(member.name))
                        folders_with_images.add(class_name)

                # Second pass: Build the class map only for folders with images
                for member in tar.getmembers():
                    if member.isdir():
                        continue  # Skip directories
                    class_name = os.path.basename(os.path.dirname(member.name))
                    if class_name in folders_with_images:
                        if class_name not in class_map:
                            class_map[class_name] = []
                        class_map[class_name].append(member.name)

        else:
            raise ValueError("Provided path is neither a valid tar file nor a directory.")

    elif os.path.isdir(tar_or_dir_path):
        logger.info("Detected directory.")
        for root, _, files in os.walk(tar_or_dir_path):
            for file in files:
                if file.lower().endswith(("jpg", "jpeg", "png")):
                    class_name = os.path.basename(root)
                    if class_name not in class_map:
                        class_map[class_name] = []
                    class_map[class_name].append(os.path.join(root, file))

    else:
        raise ValueError("Provided path is neither a valid tar file nor a directory.")

    # Create a sorted list of class names and map them to indices
    sorted_class_names = sorted(class_map.keys())
    logger.info(f"Found {len(sorted_class_names)} classes.")
    class_map = {class_name: idx for idx, class_name in enumerate(sorted_class_names)}
    logging.info("Class map: %s", class_map)
    return class_map


def extract_class_mapping_df(df: pd.DataFrame, class_col: str = "class") -> dict:
    """Creates a class mapping based on the unique values in the class column. 
    
    The mapping is created by assigning a unique integer value to each class label 
    by their alphabetical order. Adding 1 to the index to avoid 0 as a class label
    conflicting with the value for rest classes.
    
    Args:
        df: Dataframe containing the class column.
        class_col: The column containing the class labels.
        
    Returns:
        A dictionary containing the class labels and their corresponding values.
        
    Examples:
    
        Given the following dataframe:
        
        | class   |
        |---------|
        | class1  |
        | class2  |
        | class3  |
        
        The resulting class mapping would be:

        ..code-block:: python

            {
                "class1": 1,
                "class2": 2,
                "class3": 3
            }
    """
    
    logger.debug("Creating the class mapping based on the unique class labels.")

    
    class_labels = df[class_col].unique()
    logger.debug("Unique class labels found: %s", class_labels)

    return {label: idx + 1 for idx, label in enumerate(sorted(class_labels))}



def class_name_to_modul( class_name: str) -> str:
        """Converts a class name (Examplename) to modul name (example_name)

        Class names should be written in CamelCase and modul names in snake_case.
        This follows the PEP8 and Google naming conventions.

        Args:
            class_name: Class name as string in CamelCase.

        Returns:
            Converted modul name as string in snake_case.
        """

        return "_".join(
            [
                word.lower()  # lower case the word
                for word in re.findall(  # iterate over each word in the class name
                    # find all parts of the class name that start with a capital letter
                    r"[A-Z][a-z0-9]*",
                    class_name,
                )
            ]
        )

    

def import_class(class_name: str, modul_path: str = ""):
    """Imports the split strategy class from the given modul.

    Currently only works with moduls inside of the lit_ecology_classifier package.
    #TODO: Add the possibility to import moduls from other packages or from the
    # the own project.

    Args:
        modul_name: The Name of the modul to be loaded as string
        modul_path: Optional path to the modul

    Returns:
        The imported class of the modul.

    Raises:
        ModuleNotFoundError: If the modul could not be imported
        AttributeError: If the class could not be found inside the modul
    """

    logger.debug("Importing the modul %s%s", modul_path, class_name)

    # try to import the modul based on the given path and name dynamically
    modul_name = class_name_to_modul(class_name)

    # try to import the modul
    try:
        imported_modul = importlib.import_module(modul_path + modul_name)

    except ModuleNotFoundError as e:
        logger.error(
            "Modul %s not found inside the path %s: %e", modul_name, modul_path, e
        )

    except Exception as e:
        logger.error("The import of modul %s failed: %s", modul_name, e)
        raise e

    # try to get the class from the imported modul
    try:
        return getattr(imported_modul, class_name)

    except AttributeError as e:
        logger.error(
            "Class %s not found inside the modul %s", class_name, modul_name
        )
        raise e

