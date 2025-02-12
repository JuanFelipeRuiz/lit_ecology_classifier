"""
Predict on Unseen Data Using a Legacy Model
===========================================

This script predicts on images using a legacy model.

The script initializes the model, sets up the architecture, and loads the trained weights from the legacy model.
It will then predict on unseen data and save the predictions in the output directory.

Requirements
------------
The script needs the following artifacts from the legacy model:

- `params.npy`: A numpy file containing the parameters of the legacy model.
- `classes.npy`: A numpy file containing the classes of the legacy model.
- `trained_model_original.pth`: A dictionary containing the trained state weights, optimizer state, and other
  information of the legacy model.

Example
-------
To predict on unseen data using a legacy model, run the following command:

.. code-block:: bash

    python lit_ecology_classifier/legacy_predictions.py --datapath ZooLake2/Predict/unknown --main_param_path models/cheng/BEiT/ 
    --outpath ./predictions/ --model_path models/cheng/BEiT/trained_models/low_aug_01/
"""

import argparse
import pathlib
import sys
from time import time

import numpy as np
import lightning as pl
import logging
import warnings

from lit_ecology_classifier.data.datamodule import DataModule
from lit_ecology_classifier.helpers.argparser import inference_argparser
from lit_ecology_classifier.models.model import LitClassifier
time_begin = time()

# Add logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LoadModelInputParameters:
    def __init__(self, initMode='default', verbose=True):
        self.verbose = verbose
        self.params = None
        self.SetParameters(mode=initMode)

    def SetParameters(self, mode='default'):
        """ default, from args """
        if mode == 'default':
            self.ReadArgs(string=None)

        # read args from a sys.argv list
        elif mode == 'args':
        
            self.ReadArgs(string=sys.argv[1:])
        else:
            print('Unknown parameter mode', mode)
            raise NotImplementedError
        return

    def ReadArgs(self, string=None):
        if string is None:
            string = ""

        parser = argparse.ArgumentParser(description='Train a model on Zoolake2 dataset')

        parser.add_argument('-datapaths', nargs='*',
                            default=['./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/'],
                            help="Directories with the data.")
        parser.add_argument('-outpath', default='./out/', help="directory where you want the output saved")

    
        
        # Not really more needed
        parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the test set")
        parser.add_argument('-class_select', nargs='*', default=None,
                            help='List of classes to be looked at (put the class names '
                                 'one by one, separated by spaces).'
                                 ' If None, all available classes are studied.')
        parser.add_argument('-classifier', choices=['binary', 'multi', 'versusall'], default='multi',
                            help='Choose between "binary","multi","versusall" classifier')
        
        # Only needed if we still want to use the mixed data
        parser.add_argument('-ttkind', choices=['mixed', 'feat', 'image'], default=None,
                            help="Which data to use in the test and training sets: features, images, or both")
        parser.add_argument('-datakind', choices=['mixed', 'feat', 'image'], default=None,
                            help="Which data to load: features, images, or both")
        
        
        parser.add_argument('-balance_weight', choices=['yes', 'no'], default='no',
                            help='Choose "yes" or "no" for balancing class weights for imbalance classes')
        
        parser.add_argument('-training_data', choices=['True', 'False'], default='False',
                            help="This is to cope with the different directory structures")
        

        # Augmentation parameters
        parser.add_argument('-aug', action='store_true',
                            help="Perform data augmentation. Augmentation parameters are hard-coded.")
        parser.add_argument('-L', type=int, default=128, help="Images are resized to a square of LxL pixels")
        parser.add_argument('-aug_type', choices=['no', 'low', 'medium', 'high'],
                            default='low', help='Choose the augmentations intensity levels ( "low", "medium", "high")')
        parser.add_argument('-resize_images', type=int, default=1,
                            help="Images are resized to a square of LxL pixels by keeping the initial image "
                                 "proportions if resize=1. If resize=2, then the proportions are not kept but resized "
                                 "to match the user defined dimension")
        parser.add_argument('-image_size', type=int, default=224, help="Image size for training the model")

        # Data saving and loading   
        parser.add_argument('-save_data', choices=['yes', 'no'], default=None,
                            help="Whether to save the data for later use or not")
        parser.add_argument('-saved_data', choices=['yes', 'no'], default=None,
                            help="Whether to use the saved data for training")
        parser.add_argument('-compute_extrafeat', choices=['yes', 'no'], default=None,
                            help="Whether to compute extra features or not")
        parser.add_argument('-valid_set', choices=['yes', 'no'], default='yes',
                            help="Select to have validation set. Choose from Yes or No")
        parser.add_argument('-test_set', choices=['yes', 'no'], default='yes',
                            help="Select to have validation set. Choose from Yes or No")

        # Choose dataset name
        parser.add_argument('-dataset_name', choices=['zoolake', 'zooscan', 'whoi', 'kaggle',
                                                      'eilat', 'rsmas', 'birds', 'dogs', 'beetle', 'wildtrap',
                                                      'cifar10', 'inature', 'cifar100'],
                            default='zoolake', help='Choose between different datasets "zoolake", "zooscan", "whoi", '
                                                    '"kaggle", "eilat", "rsmas", "birds", "dogs", "beetle", "wildtrap"')

        # For model training
        parser.add_argument('-architecture', choices=['efficientnetb2', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'densenet', 'mobilenet', 'inception', 'deit', 'vit', 'mae', 'swin', 'beit'],
                            default='deit', help='Choose the model architecture')

        parser.add_argument('-batch_size', type=int, default=16, help="Batch size for training")
        
        parser.add_argument('-epochs', type=int, default=30, help="number of epochs for training the model")
        # parser.add_argument('-initial_epoch', type=int, default=0, help="set the initial epoch value")
        parser.add_argument('-gpu_id', type=int, default=0, help="select the gpu id ")
        parser.add_argument('-lr', type=float, default=1e-4, help="starting learning rate")
        parser.add_argument('-finetune_lr', type=float, default=1e-5, help="starting finetuning learning rate")
        parser.add_argument('-warmup', type=int, default=10, help="starting learning rate")
        parser.add_argument('-weight_decay', type=float, default=3e-2, help="weight decay")
        parser.add_argument('-clip_grad_norm', type=float, default=0, help="clip gradient norm")
        parser.add_argument('-disable_cos', choices=[True, False], default=True,
                            help="Disable cos. Choose from Yes or No")
        parser.add_argument('-run_early_stopping', choices=['yes', 'no'], default='no', )
        parser.add_argument('-run_lr_scheduler', choices=['yes', 'no'], default='no', )
        parser.add_argument('-save_best_model_on_loss_or_f1_or_accuracy', type=int, default=2,
                            help='Choose "1" to save model based on loss or "2" based on f1-score or "3" based on accu')
        parser.add_argument('-use_gpu', choices=['yes', 'no'], default='no', help='Choose "no" to run using cpu')

        # Superclass or not
        parser.add_argument('-super_class', choices=['yes', 'no'], default='yes', )
        parser.add_argument('-finetune', type=int, default=0, help='Choose "0" or "1" or "2" for finetuning')
        parser.add_argument('-finetune_epochs', type=int, default=40,
                            help="Total number of epochs for the funetune training")
        parser.add_argument('-init_name', default='Init_01',
                            help="directory name where you want the Best models to be saved")

        # Related to predicting on unseen
        parser.add_argument('-test_path', nargs='*', default=['./data/'], help="directory of images where you want to "
                                                                               "predict")
        parser.add_argument('-main_param_path', default='./out/trained_models/', help="main directory where the "
                                                                                      "training parameters are saved")
        parser.add_argument('-test_outpath', default='./out/', help="directory where you want to save the predictions")
        parser.add_argument('-model_path', nargs='*',
                            default=['./out/trained_models/Init_0/',
                                     './out/trained_models/Init_1/'],
                            help='path of the saved models')
        parser.add_argument('-finetuned', type=int, default=2, help='Choose "0" or "1" or "2" for finetuning')
        parser.add_argument('-threshold', type=float, default=0.0, help="Threshold to set")


        # TTA flags
        parser.add_argument('-TTA_type', type=int, default=0,
                            help='Choose the version of test-time augmention')
        parser.add_argument('-TTA', choices=['yes', 'no'], default='no',
                            help='Use test-time augmention or not')

        # Related to ensembling
        parser.add_argument('-ensemble', type=int, default=0,
                            help="Set this to one if you want to ensemble multiple models else set it to zero")
        parser.add_argument('-predict', type=int, default=None, help='Choose "0" for training anf "1" for predicting')

        # to run it on Google COLAB or CSCS
        parser.add_argument('-run_cnn_or_on_colab', choices=['yes', 'no'], default='no', )

        # Train from previous saved models or not
        parser.add_argument('-resume_from_saved', choices=['yes', 'no'], default='no', )
        parser.add_argument('-last_layer_finetune', choices=['yes', 'no'], default='no', )
        parser.add_argument('-last_layer_finetune_1', choices=['yes', 'no'], default='no', )
        parser.add_argument('-last_layer_finetune_2', choices=['yes', 'no'], default='no', )
        parser.add_argument('-save_intermediate_epochs', choices=['yes', 'no'], default='no', )

        # additional layers
        parser.add_argument('-add_layer', choices=['yes', 'no'], default='no')
        parser.add_argument('-dropout_1', type=float, default=0.4)
        parser.add_argument('-dropout_2', type=float, default=0.3)
        parser.add_argument('-fc_node', type=int, default=512)


        args = parser.parse_args(string)

        for i, elem in enumerate(args.datapaths):
            args.datapaths[i] = elem + '/'

        args.outpath = args.outpath + '/'
        args.training_data = True if args.training_data == 'True' else False

        self.params = args

        if self.verbose:
            print(args)

        return

    



class LoadPredictParameters:
    def __init__(self, initMode='default', verbose=True):
        self.verbose = verbose
        self.params = None
        self.SetParameters(mode=initMode)
        return

    def SetParameters(self, mode='default'):
        """ default, from args"""
        if mode == 'default':
            self.ReadArgs(string=None)
        elif mode == 'args':
            self.ReadArgs(string=sys.argv[1:])

        return

    def ReadArgs(self, string=None):
        if string is None:
            string = ""

        parser = argparse.ArgumentParser(description='Create Dataset')

        parser.add_argument('--datapath', default='./data/', help="directory where you want to predict")
        parser.add_argument('--main_param_path', default='./out/trained_models/', help="main directory where the "
                                                                                      "training parameters are saved")
        parser.add_argument('--outpath', default='./out/', help="directory where you want to save the predictions")

        parser.add_argument('--model_path', default='./out/trained_models/Init_0/',help='path of the saved models')
        
        # not yet implemented in the new model
        parser.add_argument('--ensemble', type=bool, default=False,
                            help="Set this to one if you want to ensemble multiple models else set it to zero")
        parser.add_argument('-- ', type=int, default=2, help='Choose "0" or "1" or "2" for finetuning')
        parser.add_argument('--threshold', type=float, default=0.0, help="Threshold to set")

        # from the new model 
        parser.add_argument("--no_gpu", action="store_true", help="Use no GPU for inference, default is False")
        parser.add_argument('--gpu_id', type=int, default=0, help="select the gpu id ")
        parser.add_argument("--no_TTA", action="store_true", help="Disable test-time augmentation")

        args = parser.parse_args(string)
        self.params = args
        if self.verbose:
            print(args)
        return

def legacy_predict_checker(predict_args):
    """ Check if a legacy argument is supported in the new model and transform it to the new argument"""

    if "finetuned" in predict_args:
        warnings.warn("Not implemented in the new model. The model will be loaded as it is")

    if  predict_args['ensemble']:
        raise NotImplementedError("Ensemble is not implemented in the new model")
    return predict_args

def legacy_train_checker(train_args):
    """ Check the legacy arguments of the plankiformer and transform them to the new arguments
    
    Args:
        train_args: A dictionary containing the legacy arguments of the plankiformer

    Returns:
        A dictionary containing the refactored arguments of the plankiformer

    Raises:
        NotImplementedError: If the legacy arguments are not supported in the new model
    """
    train_args = vars(train_args)
    if train_args["classifier"] in ['binary', 'versusall']:
        warnings.warn("Check if the code is still valid for binary or versusall classifier (class mapping)")
    
    if train_args["ttkind"] != 'image':
        raise NotImplementedError("Only image data is supported for the new model")

    if train_args["datakind"] !='image':
        raise NotImplementedError("Only image data is supported for the new model")
    
    if train_args["compute_extrafeat"] == 'yes':
        raise NotImplementedError("Extra features are not supported in the new model")
    
    if train_args["add_layer"] == 'yes':
        train_args["add_layer"] = True
        warnings.warn("Additional layers may not work in the new model pipeline.\
                                  Code and logic are already implemented in the model but not tested")
    elif train_args["add_layer"] == 'no':
        train_args["add_layer"] = False
    
    if train_args["save_data"] == 'yes':
        warnings.warn("Saving data is not supported like in the old model, it automatically saves the data")

    if train_args["resize_images"] == 1:
        train_args["resize_with_proportions"] = True
    else:
        train_args["resize_with_proportions"] = False

 
    if train_args["L"]:
        warnings.warn("The image size is set to 128. Resize will happen with the chosen proportions settings but will be resized afterwards resized to 224")


    elif train_args["add_layer"] == 'no':
        train_args["add_layer"] = False

    # check if key exists
    if "priority_classes" not in train_args:
        train_args["priority_classes"] = []
    if "rest_classes" not in train_args:
        train_args["rest_classes"] = []
    
    return train_args


if __name__ == '__main__':

    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Loading Testing Input parameters
    predict_args = LoadPredictParameters(initMode='args')
    
    # Create Output Directory if it doesn't exist
    pathlib.Path(predict_args.params.outpath).mkdir(parents=True, exist_ok=True)

    # Load old model input parameters from the legacy model
    legacy_train_params = LoadModelInputParameters(initMode='default') 
    legacy_train_params.params = np.load(predict_args.params.main_param_path + '/params.npy', allow_pickle=True).item()
    
    # Create a new object to store the refactored parameters of the legacy model for the new model
    refactored_train_params = LoadModelInputParameters(initMode='default')
    refactored_train_params.params = legacy_train_checker(legacy_train_params.params)
    print("Refactored training parameters: ", refactored_train_params.params)

    # Read the classes from the old model and transform them to a class map for the new model
    class_list =  np.load(predict_args.params.main_param_path + '/classes.npy')
    class_map = {class_list[i]: i for i in range(len(class_list))}

    # prepare the path to the checkpoint
    model_checkpoint_path = predict_args.params.model_path

    if ".pth" not in model_checkpoint_path:
        print("Checkpoint path: ",  model_checkpoint_path + '/trained_model_' + 'original' + '.pth')
        PATH = model_checkpoint_path + '/trained_model_' + 'original' + '.pth'
    else:
        PATH = model_checkpoint_path
    
    # overwrite the refactored train parameters to initialize the model
    refactored_train_params.params['trained_weights_path'] = PATH
    refactored_train_params.params['class_map'] = class_map
    refactored_train_params.params['datapath'] = predict_args.params.datapath
    refactored_train_params.params['outpath'] = predict_args.params.outpath
    refactored_train_params.params['TTA'] =  not predict_args.params.no_TTA
       
    

    # init a data module and dataset for prediction
    data_module = DataModule(
        datapath = refactored_train_params.params["datapath"],
        batch_size = 32,
        dataset = None,
        TTA =  refactored_train_params.params["TTA"],
        class_map = class_map,
        priority_classes = [],
        rest_classes = [],  
        resize_with_proportions = refactored_train_params.params["resize_with_proportions"],
        target_size = refactored_train_params.params["L"],
        normalize_images = False # Normalization was not implemented in the legacy model
    )

    # set up the data module to predict and do tta if needed
    data_module.setup("predict")

    # Create a new model object and load the data module
    model = LitClassifier(**refactored_train_params.params )
    model.load_datamodule(data_module)

    # define the trainer for prediction
    # Set the number of GPUs to use for prediction if no_gpu is not set
    trainer = pl.Trainer(
        devices=  1, 
        strategy= "auto",
        enable_progress_bar=False,
        default_root_dir=predict_args.params.outpath,
        limit_predict_batches= None
        )
    
  
    trainer.predict(model, datamodule=data_module)

    # Calculate and log the total time taken for prediction
    total_secs = -1 if time_begin is None else (time() - time_begin)

    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))
