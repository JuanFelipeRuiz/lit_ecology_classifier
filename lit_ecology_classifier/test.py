###########
# IMPORTS #
###########

import logging
import pathlib
import sys
from time import time
import pprint

import lightning as pl
import torch

from .data.datamodule import DataModule
from .helpers.argparser import inference_argparser
from .models.model import LitClassifier
from lit_ecology_classifier.helpers.modelling_plots import plot_confusion_matrix, plot_reduced_classes

# Start timing the script
time_begin = time()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############
# MAIN SCRIPT #
###############

def main(args):

    
    # split the path of the model to get the whole model path dict_name with apthlib
    model_folders = list(pathlib.Path(args.model_path).parts)

    logging.info("Model Path: %s", model_folders[-1].split(".")[0])
    model_folders[-1] = model_folders[-1].split(".")[0]

    model_name = "_".join(model_folders)
    # Initialize the Model

    # Create Output Directory if it doesn't exist
    pathlib.Path(args.outpath).mkdir(parents=True, exist_ok=True)
    timestamp = str(time_begin).split(".")[0]
    args.outpath = pathlib.Path(args.outpath) / f"{model_folders[-1]}_{timestamp}"
    args.outpath.mkdir(parents=True, exist_ok=True) 

    model = LitClassifier.load_from_checkpoint(args.model_path)

    # Initialize the Data Module
    hparams = model.hparams  # copy the hyperparameters from the model

    model.hparams.batch_size = args.batch_size
    model.hparams.TTA = not args.no_TTA  # set the TTA flag based on the argument
    model.hparams.outpath = args.outpath
    model.hparams.datapath = args.datapath
    model.hparams.use_wandb = False
    model.hparams.model_name = model_name
    logging.info("Parameters for the prediction:%s", pprint.pformat(model.hparams))
    # model.hparams.priority_classes = "config/priority.json" #TODO remove this
    data_module = DataModule(**model.hparams)
    data_module.setup("predict")

    model.load_datamodule(data_module)

    # Initialize the Trainer and Perform Predictions

    cudas = torch.cuda.device_count()
    trainer = pl.Trainer(devices=min(cudas, 1) if not args.no_gpu else 0,
                         strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
                         enable_progress_bar=True, default_root_dir=args.outpath)
    trainer.test(model, datamodule=data_module)

    # Calculate and log the total time taken for prediction
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))

if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Parse Arguments for prediction
    parser = inference_argparser()
    args = parser.parse_args()
    

    main(args)
