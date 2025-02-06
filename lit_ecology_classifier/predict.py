###########
# IMPORTS #
###########

import logging
import pathlib
import sys
from time import time
import pprint
import os
import json

import lightning as pl
import torch
import wandb

from .data.datamodule import DataModule
from .helpers.argparser import inference_argparser
from .models.model import LitClassifier
# Start timing the script
time_begin = time()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############
# MAIN SCRIPT #
###############


if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Parse Arguments for prediction
    parser = inference_argparser()
    args = parser.parse_args()

    

    # Create Output Directory if it doesn't exist
    pathlib.Path(args.outpath).mkdir(parents=True, exist_ok=True)

    # Initialize the Model
    model = LitClassifier.load_from_checkpoint(args.model_path)
    

    # Initialize the Data Module
    hparams = model.hparams # copy the hyperparameters from the model
    
    # Update the hyperparameters based on the given arguments
    model.hparams.batch_size = args.batch_size
    model.hparams.TTA = not args.no_TTA # If no_TTA is true, set TTA to false
    model.hparams.outpath = args.outpath
    model.hparams.datapath = args.datapath

    # Print the updated hyperparameters
    logging.info("Parameters for the prediction:%s", pprint.pformat(model.hparams))

    
    data_module = DataModule(**model.hparams)
    data_module.setup("predict")

    model.load_datamodule(data_module)

    logging.debug("DataModule setup completed")

    logging.info("Starting setup of PyTorch Lightning Trainer")
    
    
    availables_gpus = torch.cuda.device_count()
    logging.info("Available GPUs: %s", availables_gpus)
    print("Available GPUs: ", availables_gpus)
    
        # Initialize the Trainer and Perform Predictions
    logging.debug("Starting initialization of Trainer")



    # WandB initialisieren
    wandb.init(project="gpu_performance_tracking", name="predicting_with_multiple_gpus")

    trainer = pl.Trainer(

        # Set the number of GPUs to use for prediction if no_gpu is not set
        devices=  availables_gpus, 
        strategy= "auto",
        enable_progress_bar=args.prog_bar,
        default_root_dir=args.outpath,
        limit_predict_batches=args.limit_pred_batches if args.limit_pred_batches > 0 else None
        )
    
  
    trainer.predict(model, datamodule=data_module)

    # Calculate and log the total time taken for prediction
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))
