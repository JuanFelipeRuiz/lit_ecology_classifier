"""
model setup
============

This module provides a flexible interface for setting up deep learning models
with various architectures and configurations.

Features
--------
- Flexible architecture selection and simple to extend
- Download and cache model backbones from timm library
- Configure trainable parameters for full training or fine-tuning
- Add a fully connected layer to the model head


Main Components
-------------
* SetupModel : Main class for model configuration
* setup_model : Helper function to instantiate models

Notes
-----
The model backbone needs to be either cached locally or downloaded from 
huggingface. This may fail on CSCS nodes due to restricted internet access.
"""



import logging

import timm
import torch

logger = logging.getLogger(__name__)

def setup_model(
    dropout_1 = 0.4,
    dropout_2 = 0.3,
    fc_node = 512,
    add_layer = False,
    finetune = False,
    pretrained=True,
    num_classes=None,
    checkpoint_path="checkpoints/backbone.safetensors",
    architecture = "beitv2",
    **kwargs,
):
    """
    Helper function to instantiate the model setup class and return the model.
    
    
    """
    model = SetupModel(
        architecture=architecture,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        add_layer=add_layer,
        finetune=finetune,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
        fc_node=fc_node,
        pretrained=pretrained,
    ).setup_model()

    return model


        

class SetupModel:
    """
    Set up the model architecture and the trainable parameters for the model.

    Arguments:
        architecture: The base architecture / backbone of the model to be used.
        add_layer: A flag to indicate if additional layers should be added to the model.
        num_classes: The number of classes in the dataset.
        checkpoint_path: The path to the local checkpoint file with the model weights.
                        Not to be confused with the checkpoint file after training, since a regular checkpoint file of a model
                        can just be loaded with pytorch_lightning.load_from_checkpoint() 
        trained_weights_path: The path to the trained weights of the model.
        finetune: A flag to indicate if only the model head should be trained.
        dropout_1: The dropout rate for the first dropout layer.
        dropout_2: The dropout rate for the second dropout layer.
        fc_node: The number of nodes in the fully connected layer.
        pretrained: Download the model with pretrained weights.
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
        self.model = None
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

        logger.info("Model setup completed.")
        logger.info("Model architecture: %s", self.architecture)
        logger.info("Pretrained: %s", self.pretrained)
        logger.info("Number of classes: %s", self.num_classes)
        logger.info("Add additional layer: %s", self.add_layer)
        logger.info("Total number of trainable parameters: %d", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        logger.info("Total number of parameters: %d", sum(p.numel() for p in self.model.parameters()))
        return self.model

    def prepare_model_backbone(self):
        """
        Prepare the model backbone using the timm library and huggingface. 

        Prepares the model backbone from a cached file or download it from huggingface using the timm library.
        Cached files are stored per default inside the $HOME/.cache/torch/hub/checkpoints/ directory (linux).
        The model is loaded depending on the architecture, number of classes and chosen pretrained weights.
        In CSCS this may fail since the nodes have no direct internet connection. 
        #TODO: Find a solution for the CSCS nodes
        """

        
        logger.info("Preparing the model using a cached backbone or downloading from huggingface.")

        # model mapping for the timm library
        model_mapping = {
            'deit': 'deit_base_distilled_patch16_224.fb_in1k',
            'efficientnetb2': 'tf_efficientnet_b2.ns_jft_in1k',
            'efficientnetb5': 'tf_efficientnet_b5.ns_jft_in1k',
            'efficientnetb6': 'tf_efficientnet_b6.ns_jft_in1k',
            'efficientnetb7': 'tf_efficientnet_b7.ns_jft_in1k',
            'densenet': 'densenet161.tv_in1k',
            'mobilenetv3s': 'tf_mobilenetv3_small_075.in1k',
            'mobilenetv3l': 'mobilenetv3_large_100.miil_in21k_ft_in1k',
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
            logger.warning("%s not found in the model mapping.", self.architecture)

    def add_additional_layers(self):
        """
        Add additional layers to the model for fine-tuning. 
        
        Needed for the integration of legacy models. Removes the last layer of the head,
        adds one additional layer with dropout and adds the  final layer based on the number of classes.
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
        Set the trainable parameters of the model for full training or fine-tuning.
        """
        n_layer = 0

        # define if the model should be finetuned
        if self.finetune:

            # define the number of layers to unfreeze. increase number if additional layers are added
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

