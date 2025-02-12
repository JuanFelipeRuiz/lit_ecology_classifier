

import logging
from typing import Union
from pathlib import Path


import numpy as np
import timm
import torch
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


def setup_model(
    dropout_1 = 0.4,
    dropout_2 = 0.3,
    fc_node = 512,
    add_layer = False,
    finetune = False,
    pretrained=False,
    num_classes=None,
    checkpoint_path="checkpoints/backbone.safetensors",
    architecture = "beitv2",
    trained_weights_path = None,
    **kwargs,
):
    return SetupModel(
        architecture,
        checkpoint_path,
        num_classes,
        add_layer,
        finetune,
        dropout_1,
        dropout_2,
        fc_node ,
        pretrained,
        trained_weights_path,
        **kwargs,
    ).setup_model()

class SetupModel:
    """
    Set up the model architecture and the trainable parameters for the model.

    Arugments:
        architecture: The base architecture / backbone of the model to be used.
        add_layer: A flag to indicate if additional layers should be added to the model.
        num_classes: The number of classes in the dataset.
        checkpoint_path: The path to the local checkpoint file with the model weights.
                        Not to be confused with the checkpoint file after training, since a regular checkpoint file of a model
                        can just be loaded with pytorch_lightning.load_from_checkpoint() 
        trained_weights_path: The path to the trained weights of the model.
        last_layer_finetune: A flag to indicate if only the last layer should be finetuned.
        dropout_1: The dropout rate for the first dropout layer.
        dropout_2: The dropout rate for the second dropout layer.
        fc_node: The number of nodes in the fully connected layer.
        pretrained: Unused yet.
    """

    def __init__( self,
            architecture = "beitv2",
            checkpoint_path="checkpoints/backbone.safetensors",
            num_classes=None,
            add_layer = False,
            finetune = False,
            dropout_1 = 0.4,
            dropout_2 = 0.3,
            fc_node = 512,
            pretrained=True,
            trained_weights_path = None,
            **kwargs,
        ):
    
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.fc_node = fc_node
        self.add_layer = add_layer
        self.finetune = finetune
        
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.architecture = architecture
        self.trained_weights_path = trained_weights_path
        self.model = None

        # not used yet
        self.pretrained = pretrained

    def setup_model(self):
        """ Set up the model architecture and the trainable parameters for the model.

        Orchestrates the preparation of the model backbone, adding additional layers, and setting the trainable parameters.
        based on the given arguments.

        Returns:
            torch.nn.Module: The model with the prepared architecture and trainable parameters.
        """
        self.prepare_model_backbone()

        if self.add_layer:
            self.add_additional_layers()

        self.set_trainable_params()

        if self.trained_weights_path:
            self.load_trained_weights()

        logger.debug("Model setup completed.")
        logger.debug("Total number of trainable parameters: %d", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        logger.debug("Total number of parameters: %d", sum(p.numel() for p in self.model.parameters()))
        return self.model
    

    def load_trained_weights(self):
        """
        Load the state dict of the model into the prepared model.
        """
        logger.debug("Loading the trained weights of the model from %s", self.trained_weights_path)
        checkpoint = torch.load(self.trained_weights_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])


    def prepare_backbone_from_huggingface(self):
        """
        Prepare the model from a cached backbone or download from huggingface.
        using the timm library. The model is prepared with the pretrained weights and with the
        correct number of classes in the final layer/ head of the model.
        """

        # dictionary mapping model names to huggingface model names
        model_mapping = {
            'deit': 'deit_base_distilled_patch16_224.fb_in1k',
            'efficientnetb2': 'tf_efficientnet_b2.ns_jft_in1k',
            'efficientnetb5': 'tf_efficientnet_b5.ns_jft_in1k',
            'efficientnetb6': 'tf_efficientnet_b6.ns_jft_in1k',
            'efficientnetb7': 'tf_efficientnet_b7.ns_jft_in1k',
            'densenet': 'densenet161.tv_in1k',
            'mobilenet': 'mobilenetv3_large_100.miil_in21k_ft_in1k',
            'inception': 'inception_v4.tf_in1k',
            'vit': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            'mae': 'vit_base_patch16_224.mae',
            'swin': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            'beit': 'beit_base_patch16_224.in22k_ft_in22k_in1k',
            'beitv2': 'beitv2_base_patch16_224.in1k_ft_in22k'
        }

        if self.architecture in model_mapping:
            self.model = timm.create_model(model_mapping[self.architecture], pretrained= self.pretrained, num_classes=self.num_classes)
        else:
            logger.warning("%s not implemented and tested, please check the compatibility with the package.", self.architecture)
            try:
                self.model = timm.create_model(self.architecture, pretrained= self.pretrained, num_classes=self.num_classes)
            except:
                raise NotImplementedError(f"Architecture {self.architecture} could not be prepared from huggingface.")
            
    def prepare_backbone_from_local_checkpoint(self):
        """
        Set up the model from a local checkpoint file for a beitV2 architecture.

        
        Currently (2024) the slurm nodes from CSCS cant download files directly, so we make an
        extremly ugly hack. First the checkpoint, a dictionary like file with the model weights / state dict
        need to be downloaded from huggingface. Can be done manually or with get_model.sh The model is then
        initialised with random weights and the head respectively the last layerare adjusted to match the 
        number of classes in the dataset.
        """

        if self.architecture == "beitv2":

            # The slurm nodes cant download files directly currently so we make an extremly ugly hack
            # first the ckpt is download with get_model.sh, then the model is initialised with random weights
            model = timm.models.beit_base_patch16_224(pretrained=False, num_classes=1000)

            # Load the checkpoint manually
            checkpoint = load_file(self.checkpoint_path)
            model.load_state_dict(checkpoint)

            # Remove the head of the model
            # to manually add the head with the correct number of classes
            del checkpoint["head.weight"]
            del checkpoint["head.bias"]

            # Load the remaining state dict
            model.load_state_dict(checkpoint, strict=False)

            # Modify the model to match the number of classes in your dataset
            model.head = torch.nn.Linear(model.head.in_features, self.num_classes)


            self.model = model

        else:
            raise NotImplementedError(f"Architecture {self. architecture} not implemented for local checkpoint setup.")
            

    def prepare_model_backbone(self):
        """
        Prepare the model backbone from using the timm library and huggingface. We assume that the preparation from huggingface
        can fail because it is not cached or the internet connection is not available. In this case we try to prepare the model
        from a local checkpoint.

        This case happens using the CSCS slurm nodes, where the nodes have no direct internet connection. (Extremly ugly hack)
        #TODO: Find a better solution for the CSCS nodes
        """

        try:
            logger.info("Preparing the model using a cached backbone or downloading from huggingface.")
            self.prepare_backbone_from_huggingface()

        except NotImplementedError as e:
            logger.warning(e)
            
        # try to prepare the model from a local checkpoint, since it could have failed because of the missing internet connection
        except:
            logger.info("Preparing the model using a local checkpoint.")
            self.prepare_backbone_from_local_checkpoint()


    def add_additional_layers(self):
        """
        Add a additional layers to the model for fine-tuning.

        Removes the classifier head of the model and adds one additional layer (with dropout) and a final 
        layer with the number of classes.

        """

        if self.architecture == "deit":
            in_features = self.model.get_classifier()[-1].in_features
            pretrained_layers = list(self.model.children())[:-2]
        else:
            in_features = self.model.get_classifier().in_features
            pretrained_layers = list(self.model.children())[:-1]

        additional_layers = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout_1),
            torch.nn.Linear(in_features=in_features, out_features=self.fc_node),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=self.dropout_2),
            torch.nn.Linear(in_features=self.fc_node, out_features=self.num_classes),
        )
        self.model = torch.nn.Sequential(*pretrained_layers, additional_layers)

    def set_trainable_params(self):
        """
        Set the trainable parameters of the model.
        """
        n_layer = 0

        # define if the model should be finetuned
        if self.finetune:

            # define the number of layers to unfreeze. increase nubmer if additional layers are added
            layer_to_unfreeze = 2 if self.add_layer == False else 5

            # freeze all layers 
            for param in self.model.parameters():
                n_layer += 1
                param.requires_grad = False

            # unfreeze the last n layers 
            for i, param in enumerate(self.model.parameters()):
                if i + 1 > n_layer - layer_to_unfreeze:
                    param.requires_grad = True

        else:
            # unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True

