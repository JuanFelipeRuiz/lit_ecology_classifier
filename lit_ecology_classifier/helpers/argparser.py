"""
This file includes the argument parsers for the different scripts in the lit_ecology_classifier package.

The following parsers are included:
- base_argparser: Base arguments for the different scripts.
- argparser: Arguments for configuring, training, and running the machine learning model for image classification (main.py).
- inference_argparser: Arguments for using the classifier on unlabeled data (predict.py).
- overview_argparser: Arguments for creating a data overview for the given dataset (overview.py).
- split_argparser: Arguments for the split process (split.py).
"""

import argparse
import os
import json
from typing import Union

def base_argparser():
    """
    Creates an argument parser for the base arguments that are shared between the different scripts.

    Returns:
        argparse.ArgumentParser: The argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(description="Base arguments for the different scripts.")
    parser.add_argument("--dataset_name", default="phyto", help="Name of the dataset to store non train specific artifacts") 
    return parser

def argparser():
    """
    Creates an argument parser for configuring, training, and running the machine learning model for image classification.

    Arguments:
    --datapath: str
        Path to the tar file containing the training data. Default is "/store/empa/em09/aquascope/phyto.tar".
    --train_outpath: str
        Output path for training artifacts. Default is "./train_out".
    --main_param_path: str
        Main directory where the training parameters are saved. Default is "./params/".
    --dataset: str
        Name of the dataset. Default is "phyto".
    --use_wandb: flag
        Use Weights and Biases for logging. Default is False.

    --priority_classes: str
        Path to the JSON file specifying priority classes for training. Default is an empty string.
    --rest_classes: str
        Path to the JSON file specifying rest classes for training. Default is an empty string.
    --balance_classes: flag
        Balance the classes for training. Default is False.
    --batch_size: int
        Batch size for training. Default is 64.
    --max_epochs: int
        Number of epochs to train. Default is 20.
    --lr: float
        Learning rate for training. Default is 1e-2.
    --lr_factor: float
        Learning rate factor for training of full body. Default is 0.01.
    --no_gpu: flag
        Use no GPU for training. Default is False.
    --testing: flag
        Set this to True if in testing mode, False for training. Default is False.

    Returns:
        argparse.ArgumentParser: The argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(description="Configure, train and run the machine learning model for image classification.")

    # Paths and directories to use
    parser.add_argument("--datapath",  default="/store/empa/em09/aquascope/phyto.tar", help="Path to the tar file containing the training data")
    parser.add_argument("--train_outpath", default="./train_out", help="Output path for training artifacts")
    parser.add_argument("--main_param_path", default="./params/", help="Main directory where the training parameters are stored")
    parser.add_argument("--dataset", default="phyto", help="Name of the dataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights and Biases for logging")
    parser.add_argument("--no_use_multi", action="store_true", help="Use multiple GPUs for training")
    
    # Model configuration and training options
    parser.add_argument("--balance_classes", action="store_true", help="Balance the classes for training")
    parser.add_argument("--batch_size", type=int, default=180, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training")
    parser.add_argument("--lr_factor", type=float, default=0.01, help="Learning rate factor for training of full body")
    parser.add_argument("--no_gpu", action="store_true", help="Use no GPU for training, default is False")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy", help="Loss function to use")

    # Augmentation and training/testing specifics
    parser.add_argument("--testing", action="store_true", help="Set this to True if in testing mode, False for training")
    parser.add_argument("--no_TTA", action="store_true", help="Enable Test Time Augmentation")
    return parser

def inference_argparser():
    """
    Creates an argument parser for using the classifier on unlabeled data.

    Arguments:
    --batch_size: int
        Batch size for inference. Default is 180.
    --outpath: str
        Directory where predictions will be saved. Default is "./preds/".
    --model_path: str
        Path to the model checkpoint file. Default is "./checkpoints/model.ckpt".
    --datapath: str
        Path to the tar file containing the data to classify. Default is "/store/empa/em09/aquascope/phyto.tar".
    --no_gpu: flag
        Use no GPU for inference. Default is False.
    --no_TTA: flag
        Disable test-time augmentation. Default is False.
    --gpu_id: int
        GPU ID to use for inference. Default is 0.
    --limit_pred_batches: int
        Limit the number of batches to predict. Default is 0, meaning no limit, set a low number to debug.
    --prog_bar: flag
        Enable progress bar. Default is False.
    Returns:
        argparse.ArgumentParser: The argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(description="Use Classifier on unlabeled data.")
    parser.add_argument("--batch_size", type=int, default=180, help="Batch size for inference")
    parser.add_argument("--outpath", default="./preds/", help="Directory where predictions will be saved")
    parser.add_argument("--model_path", default="./checkpoints/model.ckpt", help="Path to the model checkpoint file")
    parser.add_argument("--datapath",  default="/store/empa/em09/aquascope/phyto.tar", help="Path to the tar file containing the data to classify")
    parser.add_argument("--no_gpu", action="store_true", help="Use no GPU for inference, default is False")
    parser.add_argument("--no_TTA", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for inference")
    parser.add_argument("--prog_bar", action="store_true", help="Enable progress bar")
    parser.add_argument("--limit_pred_batches", type=int, default=0, help="Limit the number of batches to predict")
    parser.add_argument("--config", type=str, default="", help="Path to the JSON file containing the configuration")
    return parser

def overview_argparser():
    """ Argparser for creating a data overview for the given dataset."""
    parser = base_argparser()

    # override description of the base parser
    parser.description = "Create an overview of the dataset."

    parser.add_argument("--overview_name", default="overview", help="Name of the overview")
    parser.add_argument("--image_version_path_dict", type=load_dict, help="Dictionary or path to the json file containing the image versions and their corresponding paths")
    parser.add_argument("--summarise", action="store_true", help="Summarise all unique images into the outputfolder")
    return parser

def split_argparser():
    """Argparser for the split process"""
    parser = argparse.ArgumentParser(description="Split the dataset into training, validation and testing sets and save the split to a file.")

    # Args for the split process
    parser.add_argument("--split_hash", type=str, default= "", help="Hash of the split to reuse. If empty, no hash search is used")
    parser.add_argument("--split_strategy", type=str, default= "", help="Split strategy to use. Needs to be saved in the lit_ecology_classifier/split_strategies folder")
    parser.add_argument("--filter_strategy", type=str, default= "", help="Filter strategy to use. Needs to be saved in the lit_ecology_classifier/filter_strategies folder")
    # Args for the split process, that can be loaded from a json file
    parser.add_argument("--split_args", type=load_dict, default= {}, help="Path to the file containing the arguments for the split strategy")
    parser.add_argument("--filter_args", type=load_dict, default= {}, help="Args or path to file containing the arguments for the filter strategy")
    parser.add_argument("--priority_classes", type= load_class_definitions, default=[], help="List of priority classes or path to the JSON file containing the priority classes")
    parser.add_argument("--rest_classes", type=load_class_definitions, default=[], help="List of rest classes or path to the JSON file containing the rest classes")

    # include the base arguments
    parser = base_argparser()
    return parser
    


def load_dict(input: Union[str, dict]) -> dict:
    """Load the training arguments from a JSON file.

    args:
        input: Path or dict containing the args.

    Returns:
        a dict containing the training arguments.

    Raises:
        argparse.ArgumentTypeError: If the input is not a dict or a exisisting path to a .json file.
    """

    if isinstance(input, dict):
        return input
    
    if input == "":
        return {}
    
    if input.endswith(".json"):
        if not os.path.exists(input):
            raise argparse.ArgumentTypeError(f"{input} file not found.")
        
        with open(input) as file:
            return json.load(file)
        
    raise argparse.ArgumentTypeError(f"{input} is not a path to a JSON file or dict containing the args.")


def load_class_definitions(input: Union[str,list]) -> list:
    """Load the the priority or rest classes from a JSON file.
    """

    if isinstance(input, list):
        return input

    if input.endswith(".json"):
        if not os.path.exists(input):
            raise argparse.ArgumentTypeError(f"{input} file not found.")
        
        with open(input) as file:
            class_dict = json.load(file)
        
        # check if priority_classes key exists
        if "priority_classes" in class_dict:
            return class_dict["priority_classes"]
        
        if "rest_classes" in class_dict:
            return class_dict["rest_classes"]
        
        raise argparse.ArgumentTypeError(f"{input} does not contain a known  class definitions.")
        
    raise argparse.ArgumentTypeError(f"{input} is not a path to a JSON file or list containing the class definitions")


# Example of using the argument parser
if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    print(args)
