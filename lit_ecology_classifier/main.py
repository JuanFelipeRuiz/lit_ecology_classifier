###########
# IMPORTS #
###########
import logging
import json
from lightning.pytorch.strategies import DDPStrategy
import pathlib
import sys
import json
from time import time

import lightning as l
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .data.datamodule import DataModule
from .helpers.argparser import argparser
from .helpers.calc_class_weights import calculate_class_weights
from .helpers.helpers import setup_callbacks
from .models.model import LitClassifier

# Start timing the script
time_begin = time()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - -%(message)s")

###############
# MAIN SCRIPT #
###############

if __name__ == "__main__":
    print("\nRunning", sys.argv[0], sys.argv[1:])

    # Parse Arguments for training
    parser = argparser()
    
    args = parser.parse_args()

    logging.info(args)

    # Create Output Directory if it doesn't exist and add a .gitignore
    pathlib.Path(args.train_outpath).mkdir(parents=True, exist_ok=True)

    with open(pathlib.Path(args.train_outpath, ".gitignore"), "w") as gitignore_file:
        gitignore_file.write("*")


    gpus =torch.cuda.device_count() if not args.no_gpu else 0
    logging.info(f"Using {gpus} GPUs for training.")
    

    # Initialize Augementation pipeline


    # Initialize the Data Module to create DataLoaders
    datamodule = DataModule(**vars(args))
    datamodule.setup("fit")

    # TODO: not implemented in main, but could be useful. Find out if the implementation is still needed and correct

    #if args.balance_classes:
    #    class_weights=calculate_class_weights(datamodule.train_dataset)
    #    models.loss = torch.nn.CrossEntropyLoss(class_weights) if not "loss" in list(models.hparams) or not models.hparams.loss=="focal" else FocalLoss(alpha=class_weights ,gamma=1.75)
    # Initialize the loggers
    
    if args.use_wandb:
        logger = WandbLogger(
            project=args.dataset,
            log_model=False,
            save_dir=args.train_outpath,
        )
        logger.experiment.log_code("./lit_ecology_classifier", include_fn=lambda path: path.endswith(".py"))
    else:
        logger = CSVLogger(save_dir=args.train_outpath, name='csv_logs')

    torch.backends.cudnn.allow_tf32 = False


    args.num_classes = len(datamodule.class_map)
    if args.balance_classes:
        args.class_weights = calculate_class_weights(datamodule)
    else:
        args.class_weights = None
    model = LitClassifier(**vars(args), finetune=True)  # TODO: check if this works on cscs, maybe add a file that downlaods model first
    model.load_datamodule(datamodule)


    definded_callbacks = [pl.callbacks.ModelCheckpoint(filename="best_model_acc_stage1", monitor="val_acc", mode="max"), LearningRateMonitor(logging_interval='step')]

    if "early_stopping" in args:
        definded_callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", delta=0.0))
    # Initialize the Trainer
    trainer = l.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        log_every_n_steps=40,
        callbacks=[pl.callbacks.ModelCheckpoint(filename="best_model_acc_stage1", monitor="val_acc", mode="max"), LearningRateMonitor(logging_interval='step')],
        check_val_every_n_epoch=max(args.max_epochs // 8,1), # Check validation every 1/8 of the max epochs or at least once
        devices=gpus,
        strategy= "ddp" if gpus > 1 else "auto" ,
        enable_progress_bar=False,
        default_root_dir=args.train_outpath,
    )
    
    # Train the first and last layer of the model
    trainer.fit(model, datamodule=datamodule)
    # Load the best model from the first stage
    model = LitClassifier.load_from_checkpoint(str(trainer.checkpoint_callback.best_model_path), lr=args.lr * args.lr_factor, pretrained=False)
    model.load_datamodule(datamodule)
    # sets up callbacks for stage 2
    callbacks = setup_callbacks(args.priority_classes, "best_model_acc_stage2")

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=2 * args.max_epochs,
        log_every_n_steps=40,
        callbacks=callbacks,
        check_val_every_n_epoch=max(args.max_epochs // 8,1),
        devices=gpus,
        strategy="ddp" if gpus > 1 else "auto",
        enable_progress_bar=False,
        default_root_dir=args.train_outpath,
    )
    trainer.fit(model, datamodule=datamodule)

    # Calculate and log the total time taken for training
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info("Time taken for training (in secs): {}".format(total_secs))
