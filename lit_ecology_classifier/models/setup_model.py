

import logging

import numpy as np
import timm
import torch
from safetensors.torch import load_file


logger = logging.getLogger(__name__)

def setup_model(
    dropout_1,
    dropout_2,
    fc_node,
    add_layer,
    last_layer_finetune,
    pretrained=False,
    num_classes=None,
    checkpoint_path="checkpoints/backbone.safetensors",
   
    **kwargs,
):
    """
    Set up and return the specified model architecture.

    Args:
        architecture (str): The model architecture to use.
        main_param_path (str): Path to the directory containing main parameters.
        ensemble (bool): Whether to use model ensembling.
        finetune (bool): Whether to finetune the model or use it as is.
        dataset (str): The name of the dataset.
        testing (bool, optional): Set to True if in testing mode. Defaults to False.
        train_first (bool, optional): Set to True to train the first layer of the model. Defaults to False.

    Returns:
        model: The configured model.
    """
    # The slurm nodes cant download files directly currently so we make an extremly ugly hack
    # first the ckpt is download with get_model.sh, then the model is initialised with random weights
    model = timm.models.beit_base_patch16_224(pretrained=False, num_classes=1000)

    # Load the checkpoint manually
    checkpoint = load_file(checkpoint_path)
    model.load_state_dict(checkpoint)

    # Remove the head
    del checkpoint["head.weight"]
    del checkpoint["head.bias"]

    # Load the remaining state dict
    model.load_state_dict(checkpoint, strict=False)

    # Modify the model to match the number of classes in your dataset
    model.head = torch.nn.Linear(model.head.in_features, num_classes)

    if add_layer:
        # add additional layers
        model = add_additional_layers(model, num_classes, **kwargs)

    # Set the trainable parameters
    set_trainable_params(
        model,
        add_layer = add_layer, 
        last_layer_finetune= last_layer_finetune
        )

    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())

    logger.info("%s total parameters.", total_params)
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info("%s training parameters.", total_trainable_params)
    print(f"{total_trainable_params:,} training parameters.")

    return model


def add_additional_layers(
    model, 
    num_classes,
    dropout_1,
    dropout_2,
    fc_node,
    architecture
):
    """
    Add additional layers to the model.

    Args:
        model (nn.Module): The model to configure.
        num_classes (int): The number of classes in the dataset.
    """ 
    logger.debug("Adding additional layers to the model.")

    if architecture == "deit":
        in_features = model.get_classifier()[-1].in_features
        pretrained_layers = list(model.children())[:-2]
    else:
        in_features = model.get_classifier().in_features
        pretrained_layers = list(model.children())[:-1]

    additional_layers = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_1),
        torch.nn.Linear(in_features=in_features, out_features=fc_node),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=dropout_2),
        torch.nn.Linear(in_features=fc_node, out_features=num_classes),
    )
    return torch.nn.Sequential(*pretrained_layers, additional_layers)


def set_trainable_params(model, add_layer, last_layer_finetune):
    """
    Set the trainable parameters of the model.

    Args:
        model (nn.Module): The model to configure.
        train_first (bool, optional): If True, train the first layer of the model. Defaults to False.
        finetune (bool, optional): If True, finetune the model. Defaults to True.
    """

    n_layer = 0

    if last_layer_finetune:

        layer_to_unfreeze = 2 if add_layer == False else 5

        for param in model.parameters():
            n_layer += 1
            param.requires_grad = False

        for i, param in enumerate(model.parameters()):
            if i + 1 > n_layer - layer_to_unfreeze:
                param.requires_grad = True

    else:
        for param in model.parameters():
            param.requires_grad = True
