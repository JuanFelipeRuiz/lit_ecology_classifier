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
import json
import logging
import os
from pathlib import Path
from typing import Union


logger = logging.getLogger(__name__)

def base_args():
    """Creates an arguemnt parser that is needed for all args.

    Returns:
        argparse.ArgumentParser: The argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("--dataset", default="phyto", help="Name of the dataset to store non train specific artifacts") 
    parser.add_argument("--overview_filename", default="overview.csv", help="Name of the overview file to load/save")
    return parser

def args_for_overview():
    """Subgroup of arguments needed for the overview creation"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_version_path_dict", type=load_dict, default="config/priority.json", help="Path to the json file containing the image versions and their corresponding paths")
    parser.add_argument("--summarise_to", type= str, default = None , help="If a path is given, the given versions are summarised int to the given path. If empty, no summarisation is done")
    return parser


def args_for_split():
     # Args for the split process
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split_hash", type=str, default= None, help="Hash of the split to reuse. If empty, no hash search is used")
    parser.add_argument("--split_strategy", type=str, default= "Stratified", help="Split strategy to use. Needs to be saved in the lit_ecology_classifier/split_strategies folder")
    parser.add_argument("--filter_strategy", type=str, default= "PlanktonFilter", help="Filter strategy to use. Needs to be saved in the lit_ecology_classifier/filter_strategies folder")
    parser.add_argument("--description", type=str, default=None, help ="Description of split, if " )
    # Args for the split process, that can be loaded from a json file
    parser.add_argument("--split_args", type=load_dict, default= None, help="Path to the file containing the arguments for the split strategy")
    parser.add_argument("--filter_args", type=load_dict, default= None, help="Path to the file containing the arguments for the filter strategy")
    parser.add_argument("--class_map", type=load_dict, default= None, help="Path to the file containing the arguments for the filter strategy")
    parser.add_argument("--priority_classes", type= load_class_definitions, default=None, help="List of priority classes or path to the JSON file containing the priority classes")
    parser.add_argument("--rest_classes", type=load_class_definitions, default=None, help="List of rest classes or path to the JSON file containing the rest classes")
    return parser

def args_for_train():
    parser = argparse.ArgumentParser(add_help = False)

    # Paths and directories to use
    parser.add_argument("--datapath",  default="/store/empa/em09/aquascope/phyto.tar", help="Path to the tar file containing the training data")
    parser.add_argument("--train_outpath", default="./train_out", help="Output path for training artifacts")
    parser.add_argument("--main_param_path", default="./params/", help="Main directory where the training parameters are stored")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights and Biases for logging")
    parser.add_argument("--no_use_multi", action="store_true", help="Use multiple GPUs for training")
    

    # Choose the model architecture
    parser.add_argument('-architecture', choices=['beitv2'],
                            default='beitv2', help='Choose the model architecture')
    
    # additional layers
    parser.add_argument('-add_layer', type=bool, default=False, help='Add additional layers to the model')
    parser.add_argument('-dropout_1', type=float, default=0.4, help='Dropout rate for the first layer')
    parser.add_argument('-dropout_2', type=float, default=0.3, help='Dropout rate for the second layer')
    parser.add_argument('-fc_node', type=int, default=512)

    # Model configuration and training options
    parser.add_argument('--last_layer_finetune', type= bool, default=False)


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

    # some additional arguments to ensure the model the old version run
    parser.add_argument("--priority_classes", type= load_class_definitions, default=[], help="Path to JSON file containing the priority classes")
    parser.add_argument("--rest_classes", type=load_class_definitions, default=[], help="Path to JSON file containing the rest classes")
    return parser


def train_argparser():
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
    parser = argparse.ArgumentParser(
        parents=[args_for_train(), base_args()],
        description="Configure, train and run the machine learning model for image classification."
    )
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
    parser = argparse.ArgumentParser(
            parents=[args_for_overview(), base_args()],
            description="Create a data overview for the given dataset."
    )
    return parser

def split_argparser():
    """Argparser for the split process"""
    parser = argparse.ArgumentParser(
        parents=[args_for_split(), base_args()],
        description="Split the data into train, validation, and test sets."
    )
    return parser

def pipeline_argparser():
    parser = argparse.ArgumentParser(
        parents=[ args_for_split(), args_for_train(),  base_args()],
        conflict_handler="resolve",
        description="Configure, train and run the machine learning model for image classification with split creation."
    )

    return parser

def argparser():
    parser = argparse.ArgumentParser(
        parents=[ args_for_train(),  base_args()],
        description="Configure, train and run the machine learning model for image classification."
    )
    return parser

def load_dict(user_input: Union[str, None]) -> dict:
    """Load the training arguments from a JSON file.

    args:
        input: Path to the args.

    Returns:
        a dict containing the training arguments.

    Raises:
        argparse.ArgumentTypeError: If the input is not a dict or a exisisting path to a .json file.
    """

    if user_input == "" or user_input is None:
        return {}
    
    user_input = Path(user_input).as_posix()
    print(user_input)
    if user_input.endswith(".json"):
        if not os.path.exists(user_input):
            raise argparse.ArgumentTypeError(f"{user_input} file not found.")
        
        with open(user_input) as file:
            return json.load(file)
        
    raise argparse.ArgumentTypeError(f"{user_input} is not a path to a JSON file or dict containing the args.")

def load_class_definitions(user_input: Union[str, None, list]) -> list :
    """Load the the priority or rest classes from a JSON file.
    """

    if user_input == "" or input is None or user_input == []:
        return []
    
    user_input = Path(user_input).as_posix() 
    
    if user_input.endswith(".json"):
        if not os.path.exists(user_input):
            raise argparse.ArgumentTypeError(f"{user_input} file not found. Searching file from {os.getcwd()}")
        
        with open(user_input) as file:
            class_dict = json.load(file)
        
        # check if priority_classes key exists
        if "priority_classes" in class_dict:
            return class_dict["priority_classes"]
            
        
        if "rest_classes" in class_dict:
            return class_dict["rest_classes"]
        
        raise argparse.ArgumentTypeError(f"{user_input} does not contain a valid class definition. Please provide a JSON file containing 'priority_classes' or 'rest_classes' as key")

    
    raise argparse.ArgumentTypeError(f"{user_input} is not a path to a JSON file or list containing the class definitions")
    
# Example of using the argument parser
if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    print(args)
