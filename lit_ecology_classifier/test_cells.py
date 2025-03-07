###########
# IMPORTS #
###########

import logging
import pathlib
import sys
from datetime import datetime
import pprint
import os

import pandas as pd
import lightning as pl
from lightning import Trainer
import torch

from lit_ecology_classifier.data.datamodule import DataModule
from lit_ecology_classifier.helpers.argparser import ood_argparser
from lit_ecology_classifier.models.model import LitClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Testing():
    """
    Test the given models with the given test datasets.

    Args:
        model_paths: list of paths to the model checkpoints
        output_path: path to the output directory
        test_cells: Optional, choose either test cells or ood dir. List of paths to the test cells. 
        ood_dir: Optional, path to the directory containing the test cells.
        batch_size: batch size for the testing
        TTA: whether to use test-time augmentation
        only_test: Only set to False if you created a model with a datamodule that reads the data from a folder or tar file and you dont have the test cell.
    
    """
    def __init__(
            self,
            model_paths ,
            output_path,
            test_cells = None,
            ood_dir = None,
            batch_size = 32,
            TTA = False,
            only_test = True,
            strict=False
    ):
        self.model_paths = model_paths
        self.output_path = self.prepare_output_dir(output_path)
        self.cells = test_cells if test_cells else self.get_ood_cells(ood_dir)
        self.batch_size = batch_size
        self.TTA = TTA
        self.model = None
        self.pl_trainer = None
        self.performance = []
        self.only_test = only_test
        self.strict = strict

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
        """
        Setup the model to be tested.
        """
        model = LitClassifier.load_from_checkpoint(model_path, strict=self.strict)
        model_name = str(model_path).split(os.sep)[-1].split(".")[0]
        model.hparams.batch_size = self.batch_size
        model.hparams.TTA = self.TTA
        model.hparams.test_outpath = self.output_path
        model.hparams.use_wandb = False
        model.hparams.model_name = model_name
        model.hparams.only_test = self.only_test
        self.model = model

    def test_cell(self, test_cell):
        """
        Test a single cell
        """
        self.model.hparams.datapath = test_cell
        data_module = DataModule(**self.model.hparams)
        data_module.setup("test")
        self.model.load_datamodule(data_module)
        self.pl_trainer.test(self.model, datamodule=data_module)
        f1 = self.model.test_f1
        accuracy = self.model.test_acc
        model = self.model.hparams.model_name
        test_cell = self.model.test_cell
        self.performance.append([model, test_cell, accuracy, f1])

    def test(self):
        """ Test each model on each cell"""
        #TODO: Provide a way to parallelize the prediction process if possible
        #TODO: Check if this function is even necessary, may be better to do it with slurm array jobs

        # loop over each model
        for model_path in self.model_paths:
            self.setup_model(model_path=model_path)
            # predict on each cell
            for cell in self.cells:
                self.test_cell(cell)

    def get_performance_overview(self):
        """
        Get the performance overview of the models on the cells
        """

        columns = ["model", "cell", "accuracy", "f1"]
        self.performance = pd.DataFrame(self.performance, columns=columns)
        return self.performance


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
    test.test()
   