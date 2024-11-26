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


    # combine model hyperparameters with the arguments from the command line to save the updated hyperparameters
    combined_hparams = {
        "model": model.hparams, 
        "predict": vars(args)  
    }

    # Print the combined hyperparameters
    logging.info("Combined Parameters:%s", pprint.pformat(combined_hparams))

    date = time()
    model_name = os.path.basename(combined_hparams["predict"]["model_path"].split(".")[0])


    # save the combined hyperparameters to a file in the output directory
    with open(args.outpath + f"/{model_name}_{date}.json", "w") as f:
        json.dump(combined_hparams, f)
    


   
    
    # Update the hyperparameters based on the given arguments
    model.hparams.batch_size = args.batch_size
    model.hparams.TTA = not args.no_TTA # set the TTA flag based on the argument
    model.hparams.outpath = args.outpath
    model.hparams.datapath = args.datapath

    # Print the updated hyperparameters
    logging.info("Parameters for the prediction:%s", pprint.pformat(model.hparams))
    data_module = DataModule(**model.hparams)
    data_module.setup("predict")

    model.load_datamodule(data_module)

    logging.debug("DataModule setup completed")

    # Initialize the Trainer and Perform Predictions
    logging.debug("Starting initialization of Trainer")
    trainer = pl.Trainer(

        # Set the number of GPUs to use for prediction if no_gpu is not set
        devices= [args.gpu_id] if not args.no_gpu else None, 
        
        strategy= "auto",
        enable_progress_bar=args.prog_bar,
        default_root_dir=args.outpath,
        limit_predict_batches=args.limit_pred_batches if args.limit_pred_batches > 0 else None
        )
    print("1")
    trainer.predict(model, datamodule=data_module)

    # Calculate and log the total time taken for prediction
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))
