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

from lit_ecology_classifier.data.datamodule import DataModule
from lit_ecology_classifier.helpers.argparser import ood_argparser
from lit_ecology_classifier.models.model import LitClassifier
from lit_ecology_classifier.helpers.modelling_plots import plot_confusion_matrix, plot_reduced_classes

# Start timing the script
time_begin = time()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_cells(ood_dir):
    """
    Get the list of cells to predict on. Each subdirectory in the directory
    is considered as a separate cell.
    """
    ood_dir = pathlib.Path(ood_dir)
    return [str(x) for x in ood_dir.iterdir() if x.is_dir()]



def main(args, cells):

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

    model.hparams.batch_size = args.batch_size
    model.hparams.TTA = not args.no_TTA  # set the TTA flag based on the argument
    model.hparams.outpath = args.outpath
    model.hparams.use_wandb = False
    model.hparams.model_name = model_name
    model.hparams.ood = True
    logging.info("Parameters for the prediction:%s", pprint.pformat(model.hparams))
    
    cudas = torch.cuda.device_count()


    for ood_cell in cells:
        logging.info("Predicting for cell: %s", ood_cell)
        model.hparams.datapath = ood_cell
        data_module = DataModule(**model.hparams)
        model.load_datamodule(data_module)
        trainer = pl.Trainer(
        devices=min(cudas, 1) if not args.no_gpu else 0,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        enable_progress_bar=True, default_root_dir=args.outpath
        )
        trainer.test(model, datamodule=data_module)


    # Calculate and log the total time taken for prediction
    total_secs = -1 if time_begin is None else (time() - time_begin)

if __name__ == '__main__':
    args = ood_argparser().parse_args()
    cells = get_cells(args.ood_dir)
    main(args, cells)