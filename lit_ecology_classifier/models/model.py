import logging
import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics 
import pandas as pd
from datetime import datetime
from lightning import LightningModule


from lit_ecology_classifier.helpers.modelling_plots import plot_confusion_matrix, plot_loss_acc, plot_score_distributions, compute_roc_auc_binary, barplot_predictions
from lit_ecology_classifier.helpers.helpers import CosineWarmupScheduler, gmean, output_results, FocalLoss, setup_classmap, test_output_results
from lit_ecology_classifier.models.setup_model import setup_model
from lit_ecology_classifier.models.metrics import compute_metrics, compute_roc_auc

class LitClassifier(LightningModule):
    def __init__(self, **hparams):
        """
        Initialize the LitClassifier.
        Args:
            hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()

        if 'class_map' not in self.hparams or self.hparams.class_map == {}:
            self.hparams.class_map = setup_classmap(datapath=self.hparams['datapath'], priority_classes=self.hparams['priority_classes'], rest_classes=self.hparams['rest_classes'])
            self.hparams.class_map = self.hparams.class_map

        self.hparams.num_classes = len(self.hparams.class_map.keys())
        self.inverted_class_map = dict(sorted({v: k for k, v in self.hparams.class_map.items()}.items()))
        self.model = setup_model(**self.hparams)
        self.loss = self.define_loss()
        self.train_metrics, self.val_metrics = self.define_metrics()

        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.hparams.model_name = f"{self.hparams.architecture}_{time_stamp}" if hasattr(self.hparams, "architecture") else f"model_{time_stamp}"
        
        logging.info("Model initialized with hyperparameters:\n {}".format(pprint.pformat(self.hparams)))

       
    def define_metrics(self):
        """
        Defines the calculated metrics for the model for training and validation.
        """

        # Define the task type based on the number of classes
        task_type = "multiclass" if self.hparams.num_classes > 2 else "binary"

        train_metrics = torchmetrics.MetricCollection({
            "acc": torchmetrics.Accuracy(num_classes=self.hparams.num_classes, task= task_type),
            "f1": torchmetrics.F1Score(num_classes=self.hparams.num_classes, average="weighted", task= task_type),
            "balanced_acc": torchmetrics.Accuracy(num_classes=self.hparams.num_classes, average="macro", task= task_type)
        }, prefix="train_")

        val_metrics = torchmetrics.MetricCollection({
            "acc": torchmetrics.Accuracy(num_classes=self.hparams.num_classes, task= task_type),
            "f1": torchmetrics.F1Score(num_classes=self.hparams.num_classes, average="weighted", task= task_type),
            "balanced_acc": torchmetrics.Accuracy(num_classes=self.hparams.num_classes, average="macro", task= task_type),
            "precision": torchmetrics.Precision(num_classes=self.hparams.num_classes, average="macro", task= task_type)
        }, prefix="val_")

        return train_metrics, val_metrics


    def define_loss(self):
        """
        Define the loss function to use for training.
        
        If the loss is set to "focal", use the FocalLoss class with gamma=1.75.
        If class weights are provided, use the CrossEntropyLoss with the class weights.
        Otherwise, use the standard CrossEntropyLoss.
        """
        loss_type = getattr(self.hparams, "loss", None)
        class_weights = getattr(self.hparams, "class_weights", None)

        if loss_type == "focal":
            if class_weights is not None:
                raise ValueError("Focal loss cannot be used with class weights.")
            logging.info("Using FocalLoss")
            return FocalLoss(alpha=None, gamma=1.75)

        if class_weights is not None:
            logging.info("Using CrossEntropyLoss with class weights.")
            return torch.nn.CrossEntropyLoss(weight=class_weights)

        logging.info("Using CrossEntropyLoss")
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        """
        # Use the AdamW optimizer with the learning rate specified in the hyperparameters
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

        # Use the CosineWarmupScheduler with a warmup period of 3 epochs and a total of max_epochs epochs
        scheduler = CosineWarmupScheduler(optimizer, warmup=3 * len(self.datamodule.train_dataloader()), max_iters=self.trainer.max_epochs * len(self.datamodule.train_dataloader()))
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]

    def TTA(self, batch):
        """
        Perform Test Time Augmentation (TTA) on the input batch.

        Args:
            batch (tuple): Input batch containing the rotation as key and the probability as value.

        Returns:
            torch.Tensor: Geometrics Average of probabilities from the TTA predictions.
            torch.Tensor: True labels if batch is list containg true labels as second entry else None.
        """

        # extract the probabilities from each rotation
        x = torch.cat([batch[str(i * 90)] for i in range(4)], dim=0)

        # use a softmax function to convert the logits to probabilities
        logits = self(x).softmax(dim=1)

        logits = torch.stack(torch.chunk(logits, 4, dim=0))

        # calculate the geometric mean of the probabilities
        logits = gmean(logits, dim=0)
        return logits

    def forward(self, x):
        """
        Forward pass through the model.
        """
        return self.model(x)

    def load_datamodule(self, datamodule):
        """
        Load the data module into the model to save the necessary hyperparameters
        for a new run.
        """
        self.datamodule = datamodule
        self.hparams.TTA = datamodule.TTA
        self.hparams.datapath = datamodule.datapath
        self.hparams.splits = datamodule.splits
        self.hparams.batch_size = datamodule.batch_size

    def training_step(self, batch, batch_idx):
        """
        Perform a training step and log the calculated loss and metrics.

        Args:
            batch (tuple): Input batch containing images and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        x, y = batch
        logits = self(x)
        
        # compute the loss 
        loss = self.loss(logits, y)

        # compute the train metrics with the predfined metrics inside of define_metrics
        self.train_metrics.update(logits, y)
        train_metrics = self.train_metrics.compute()

        # log the loss and metrics
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(train_metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        
        # reset the metrics at the end of the epoch
        self.train_metrics.reset()
        return super().on_train_epoch_end()
        

    def on_validation_epoch_start(self):
        self.val_step_predictions = []
        self.val_step_targets = []
        self.val_step_probs = []

    def validation_step(self, batch, batch_idx):
        """Perform a validation step.
        
        - Calculate the probabilities with TTA if enabled.
        - Calculate the loss.
        - Update the validation metrics.
        - Log the loss and metrics.
        - Append the predictions, probabilities, and the true label for later use.
        """
        if self.hparams.TTA:
            # calculation of the probabilities with TTA
            probs = self.TTA(batch[0])
            logits=probs
            y=batch[1]

        else:
            # normal calculation of the probabilities
            x, y = batch
            logits = self(x)
            probs=logits.softmax(dim=1)

        loss = self.loss(logits, y)

        # compute the validation metrics
        self.val_metrics.update(probs, y)
        step_metrics = self.val_metrics.compute()

        # log the loss and metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(step_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # append the predictions, probabilities, and the true label for later use
        self.val_step_probs.append(probs)
        self.val_step_predictions.append(probs.argmax(dim=1))
        self.val_step_targets.append(y)

        return loss

    def on_validation_epoch_end(self):
        """
        - Log the metrics and plots at the end of the validation epoch.
        - Plot the confusion matrix and score distributions.
        """
        # log the metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        all_scores = torch.cat(self.val_step_probs)
        all_preds = torch.cat(self.val_step_predictions)
        all_labels = torch.cat(self.val_step_targets)


        # create core for distribution of the probabilities and the true labels
        fig_score = plot_score_distributions(all_scores, all_preds, self.inverted_class_map.values(), all_labels)
        fig1,fig2, confusion_matrix, confusion_matrix_norm  = plot_confusion_matrix(all_labels, all_preds, self.inverted_class_map.values())

        del confusion_matrix, confusion_matrix_norm
        # Log the confusion matrix to wandb if use_wandb is true
        if self.hparams.use_wandb:
            self.logger.log_image(key=f"score_distributions", images=[fig_score], step=self.current_epoch)
            self.logger.log_image(key="confusion_matrix", images=[fig1], step=self.current_epoch)
            self.logger.log_image(key="confusion_matrix_norm", images=[fig2], step=self.current_epoch)
        else:
            
            # check if the logger is csvlogger
            if self.trainer.logger.__class__.__name__ == "CSVLogger":                
                log_dir = Path(self.trainer.logger.log_dir)
                confusion_matrix_epoch_path = log_dir / f"confusion_matrix_epoch_{self.current_epoch}.png"
                confusion_matrix_norm_epoch_path = log_dir  / f"confusion_matrix_normalized_epoch_{self.current_epoch}.png"
                score_distributions_epoch_path = log_dir  / f"score_distributions_epoch__{self.current_epoch}.png"
                fig1.savefig(confusion_matrix_epoch_path)
                fig2.savefig(confusion_matrix_norm_epoch_path)
                fig_score.savefig(score_distributions_epoch_path)
            
            elif self.current_epoch == self.trainer.max_epochs - 1:
                    self.train_outpath = Path(self.hparams.train_outpath) / self.hparams.model_name
                    Path.mkdir(self.train_outpath, exist_ok=True)
                    confusion_matrix_epoch_path = self.train_outpath / f"confusion_matrix_epoch_{self.current_epoch}.png"
                    confusion_matrix_norm_epoch_path = self.train_outpath  / f"confusion_matrix_normalized_epoch_{self.current_epoch}.png"
                    score_distributions_epoch_path = self.train_outpath  / f"score_distributions_epoch__{self.current_epoch}.png"
                    fig1.savefig(confusion_matrix_epoch_path)
                    fig2.savefig(confusion_matrix_norm_epoch_path)
                    fig_score.savefig(score_distributions_epoch_path)

            if self.current_epoch == self.trainer.max_epochs - 1:
                plt.show(fig1)
                plt.show(fig2)
                plt.show(fig_score)
               

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig_score)


    def on_fit_end(self) -> None:
        """
        If the model is not using wandb, plot the loss and accuracy curves at the end of training
        and save them in the output folder.
        """
        if not self.hparams.use_wandb:
            plot_loss_acc(self.trainer.logger)
            plt.show(plot_loss_acc)
            print(f"Artefatcs saved to {self.trainer.log_dir}")
            
        return super().on_fit_end()    

    def on_test_epoch_start(self):
        """
        Hook to be called at the start of the test epoch.
        Sets up empty lists to store the predicted class probabilities and filenames.
        """
        self.test_step_predictions = []
        self.test_step_targets = []
        self.test_step_probs = []
        self.test_step_highest_probs = []
        self.model.eval()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        """
        Perform a test step.
        Args:
            batch (tuple): Input batch containing images and filenames.
            batch_idx (int): Batch index.
        """

        with torch.no_grad():
            if self.hparams.TTA:
                probs = self.TTA(batch[0])
                y=batch[1]
            else:
                x,y = batch
                logits = self(x)
                probs=logits.softmax(dim=1)

            self.test_step_targets.append(y.cpu()) # Append the true label
            self.test_step_predictions.append(probs.argmax(1).cpu()) # Append the predicted label
            self.test_step_probs.append(probs.cpu()) # Append the predicted probabilities
            self.test_step_highest_probs.append(probs.max(1).values.cpu()) # Append the highest probability

    def on_test_epoch_end(self):
        """
        Aggregate outputs and log metrics and plots at the end of the test epoch.
        """
        # self.datamodule.test_dataset.dataset.image_infos
     
        if hasattr(self.datamodule.test_dataset, "image_infos"):
            filenames = self.datamodule.test_dataset.image_infos
        elif hasattr(self.datamodule.test_dataset, 'dataset') and hasattr(self.datamodule.test_dataset.dataset, 'image_infos'):
            filenames = self.datamodule.test_dataset.dataset.image_infos
        all_scores = torch.cat(self.test_step_probs)  # All predicted probabilities for each image in the test set
        all_pred_label = torch.cat(self.test_step_predictions) # Label of the class with the highest probability 
        all_y_labels = torch.cat(self.test_step_targets) # True label of the images
        all_highest_scores = torch.cat(self.test_step_highest_probs) # Highest probability for each image

        class_names = list(self.inverted_class_map.values())

        fig_score = plot_score_distributions(
            all_scores, all_pred_label, class_names, all_y_labels
        )

        balanced_acc, false_positives, precision, recall, f1, accuracy, all_y_labels_np, all_predicted_labels_np,cls_report  = compute_metrics(all_y_labels, all_pred_label, self.inverted_class_map)
        
        # Plot confusion matrices
        fig1,fig2, confusion_matrix, confusion_matrix_norm = plot_confusion_matrix(
            all_y_labels, all_predicted_labels_np, class_names
        )

        # transform the matrix (a array) to a pandas dataframe
        confusion_matrix_df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        confusion_matrix_norm_df = pd.DataFrame(confusion_matrix_norm, index=class_names, columns=class_names)

        
        # rename each column
        confusion_matrix_df.columns = [f'Predicted {col}' for col in confusion_matrix_df.columns]
        confusion_matrix_norm_df.columns = [f'Predicted {col}' for col in confusion_matrix_norm_df.columns]

        # name the first column and index
        confusion_matrix_df.index.name = 'True label'
        confusion_matrix_norm_df.index.name = 'True label'

        # Compute ROC curves and AUC
        roc_auc = compute_roc_auc(all_y_labels, all_scores)

        roc_auc_binary = compute_roc_auc_binary(all_y_labels, all_scores)

        inverted_predicted_labels_np = np.array([self.inverted_class_map[label] for label in all_predicted_labels_np])
        inverted_y_labels_np = np.array([self.inverted_class_map[label] for label in all_y_labels_np])

        column_names = ["img","true_label","predicted_label","score_of_predicted_label", "all_scores"]
        classifications = zip(filenames, inverted_y_labels_np, inverted_predicted_labels_np, all_highest_scores.numpy() , all_scores.numpy())
        
        #classifications = test_output_results(im_names=filenames, true_labels=all_y_labels, predicted_labels=inverted_labels_np, scores=all_scores)

        self.test_f1 = f1
        self.test_acc = accuracy
        self.test_cell = Path(self.hparams.datapath).name

        if self.hparams.use_wandb:
            self.logger.log_image(key=f"score_distributions", images=[fig_score], step=self.current_epoch)
            self.logger.log_image(key="confusion_matrix", images=[fig1], step=self.current_epoch)
            self.logger.log_image(key="confusion_matrix_norm", images=[fig2], step=self.current_epoch)
            
        
        else:
            try:
                base_outpath = Path(self.hparams.test_outpath) / f"{self.hparams.model_name}_{self.test_cell }"
                
            except AttributeError:
                raise AttributeError("No test_outpath specified. Saving classification artefacts to the current directory")
            Path.mkdir(base_outpath, exist_ok=True)
            path_score_distributions = base_outpath / "score_distributions.png"
            path_confusion_matrix =  base_outpath /  "confusion_matrix.png"
            path_confusion_matrix_norm =  base_outpath / "confusion_matrix_normalized.png"
            path_classification_report = base_outpath / f"classification_report_{self.hparams.model_name}_{self.test_cell}.txt"
            path_classifications = base_outpath / f"classifications_{self.hparams.model_name}_{self.test_cell}.csv"
            path_confusion_matrix_csv = base_outpath / f"confusion_matrix_{self.hparams.model_name}_{self.test_cell}.csv"
            


            fig_score.savefig(path_score_distributions)
            fig1.savefig(path_confusion_matrix)
            fig2.savefig(path_confusion_matrix_norm)
            
            cls_report_content = f"Accuracy: {accuracy}\nF1: {f1}\n\n\n{cls_report}"
            with open(path_classification_report, "w") as f:
                f.write(cls_report_content)


            df = pd.DataFrame(classifications, columns=column_names)

            df.to_csv(path_classifications, index=False)
            confusion_matrix_df.to_csv(path_confusion_matrix_csv)

        return super().on_test_epoch_end()
            

    def on_predict_start(self) -> None:
        """
        Hook for the start of the prediction process.
        """
        self.probabilities = []
        self.model.eval()

        return super().on_predict_start()

    def predict_step(self, batch) -> None:
        """
        Perform a prediction step on unlabeled data.
        Args:
            batch (tuple): Input batch containing images
        """
        with torch.no_grad():
            
            # if TTA is enabled, call the TTA method to calculate the mean of the predictions
            if self.hparams.TTA:
                probs = self.TTA(batch).cpu()

            else:
                batch = batch
                probs = self(batch).softmax(dim=1).cpu()
                
            self.probabilities.append(probs)

    def on_predict_epoch_end(self) -> None:
        """
        Hook for the end of the prediction process.
        """
        filenames = self.datamodule.predict_dataset.image_infos
        max_index = torch.cat(self.probabilities).argmax(axis=1)
        pred_label = np.array([self.inverted_class_map[idx] for idx in max_index.numpy()], dtype=object)
        pred_score = torch.cat(self.probabilities).max(1)[0].numpy()
        output_results(self.hparams.outpath, 
                        filenames, 
                        pred_label, 
                        pred_score, 
                        priority_classes=self.hparams.priority_classes!=[],
                        rest_classes=self.hparams.rest_classes!=[],
                        datapath = self.hparams.datapath,
                        legacy = True
                        )
        
        
        barplot_predictions(pred_label, self.inverted_class_map, self.hparams.outpath)
        return super().on_predict_epoch_end()

