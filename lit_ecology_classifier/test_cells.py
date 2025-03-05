###########
# IMPORTS #
###########

import logging
import pathlib
import sys
from datetime import datetime
import pprint
import os

import lightning as pl
from lightning import Trainer
import torch

from lit_ecology_classifier.data.datamodule import DataModule
from lit_ecology_classifier.helpers.argparser import ood_argparser
from lit_ecology_classifier.models.model import LitClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Testing():
    def __init__(
            self,
            model_paths ,
            output_path,
            test_cells = None,
            ood_dir = None,
            batch_size = 32,
            TTA = False,
    ):
        self.model_paths = model_paths
        self.output_path = self.prepare_output_dir(output_path)
        self.cells = test_cells if test_cells else self.get_ood_cells(ood_dir)
        self.batch_size = batch_size
        self.TTA = TTA
        self.model = None
        self.pl_trainer = None
        self.cls_reports = []

    def trainer(self, pl_trainer):
        """
        provide a pl trainer object to the class
        """
        if not isinstance(pl_trainer, Trainer):
            raise ValueError("pl_trainer should be a pl.Trainer object")
        
        self.pl_trainer = pl_trainer

    def get_ood_cells(self, ood_dir):
        """
        Get the list of cells to predict on. Each subdirectory in the directory
        is considered as a separate cell.
        """
        ood_dir = pathlib.Path(ood_dir)
        return [str(x) for x in ood_dir.iterdir() if x.is_dir()]

    def prepare_output_dir(self, outpath):
        """
        Prepare the output directory for the predictions.
        """
        # get current date and time in datetime format: '2021-09-01_12-00-00'
        time_begin = datetime.now()
        time_begin = time_begin.strftime("%Y-%m-%d_%H-%M-%S")
        outpath = pathlib.Path(outpath) / f"{time_begin}"
        outpath.mkdir(parents=True, exist_ok=True)
        return outpath

    def setup_model(self, model_path):
        """Setup the model for prediction."""
        model = LitClassifier.load_from_checkpoint(model_path)
        model_name = model_path.split(os.sep)[-1].split(".")[0]
        model.hparams.batch_size = self.batch_size
        model.hparams.TTA = self.TTA
        model.hparams.outpath = self.output_path
        model.hparams.use_wandb = False
        model.hparams.model_name = model_name
        model.hparams.only_test = True
        self.model = model

    def test_cell(self, test_cell):
        logging.info("Predicting for cell: %s", test_cell)
        self.model.hparams.datapath = test_cell
        data_module = DataModule(**self.model.hparams)
        self.model.load_datamodule(data_module)
        self.pl_trainer.test(self.model, datamodule=data_module)
        self.cls_reports.append(self.model.cls_report)

    def loop_over_cells(self):
        #TODO: Provide a way to parallelize the prediction process if possible
        for model_path in self.model_paths:
            self.setup_model(model_path=model_path)
            for cell in self.cells:
                self.test_cell(cell)

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


if __name__ == '__main__':
    #args = ood_argparser().parse_args()
    #cells = get_cells(args.ood_dir)
    #main(args, cells)

    model_paths = ["models/benno/zoo3.ckpt"]
    ood_dir = "data/mini_OOD"

    output_path = "output_test"

    test = Testing(model_paths, output_path, ood_dir=ood_dir)

    trainer = pl.Trainer(
        devices=1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        enable_progress_bar=True, default_root_dir=output_path
        )
    
    test.trainer(trainer)
    test.loop_over_cells()
    print(test.cls_reports)