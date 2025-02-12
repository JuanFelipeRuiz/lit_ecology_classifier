"""
Image transformations 
====================

Modul that defines the common image transformations for the diffrent custom 
`datasets` used in the lit_ecology_classifier package.

Classes:
    - ResizeWithProportions: Callable that resize an image while maintaining its proportions.

Functions:
    - define_resize_transform: Defines the usage of the default resize transformation of pytorch or 
        the custom resize with proportions transformation based on the given flag.
    - define_transformations: Defines which level of transformations to use based on the given 
        flag and the target size.[None, "low", "medium", "high"]
"""

from typing import Union
import warnings

from PIL import Image
import torch
from torchvision.transforms.v2 import (
    AugMix,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
)
import torchvision.transforms as T


class ResizeWithProportions:
    """
    Resizes an image while maintaining its proportions by placing it into a black square of the target size, simualting
    a padding effect.This transformation should be used before converting the image to a tensor, since it does not return a tensor
    and cannot handle possible scaling of the pixel values.

    If either dimension of the image exceeds the target size, it is resized while maintaining the aspect ratio.
    The resized image is then centered on a black square of the target size.
    If one dimension is more than 5 times the other, a warning is raised to prevent excessive shrinking.

    Attributes:
        target_size: The desired size of the output image.
        height: The height of the input image.
        width: The width of the input image.
    """

    def __init__(self, target_size: Union[tuple[int, int], int]):
        """
        Initializes the ResizeWithProportions transformation.

        Args:
            target_size: The desired size of the output image.
        """
        self.target_size = self._transform_target_size(target_size)
        self.image_height = None
        self.image_width = None

    def _transform_target_size(self, target_size: Union[tuple[int, int], int]) -> int:
        """
        Transform the input to a single integer target size.

        Args:
            target_size : The desired size of the output image.

        Returns:
            The target size as an integer.

        Raises:
            NotImplementedError: If the input is a tuple with two different values.
        """

        # check if tuple contains two similar values to extract the target size
        if isinstance(target_size, tuple):
            if target_size[0] != target_size[1]:
                raise NotImplementedError("Only square images are currently supported.")
            return target_size[0]

        # check if the input is an integer
        if isinstance(target_size, int):
            return target_size

        raise ValueError(
            f"Expected an integer or a tuple of two integers, but got {type(target_size)}"
        )

    def _check_dimensions(self):
        """
        Check if the image dimensions exceed a 5:1 ratio.

        Raises:
            Warning: If the image dimensions exceed the maximum allowed ratio of 5:1.
        """

        if self.image_width and self.image_height:
            largest_dim = max(self.image_width, self.image_height)
            smallest_dim = min(self.image_width, self.image_height)
            if (self.target_size * smallest_dim) / largest_dim < 5:
                warnings.warn(
                    "The image dimensions exceed the maximum allowed ratio of 5:1"
                )

    def _shrink_image(self, image: Image.Image) -> Image.Image:
        """
        Resizes the image to fit within the target size while maintaining aspect ratio.

        Args:
            image: The input image to resize.

        Returns:
            The resized image.
        """
        _ratio = float(self.target_size) / max(self.image_width, self.image_height)
        _new_size = (int(self.image_width * _ratio), int(self.image_height * _ratio))
        return image.resize(_new_size, Image.LANCZOS)

    def _add_padding(self, image: Image.Image) -> Image.Image:
        """
        Centers the resized image on a black square of the target size to maintain the aspect ratio.

        Args:
            image: The resized image.

        Returns:
            The padded/ centered image.
        """
        new_im = Image.new("RGB", (self.target_size, self.target_size), color=(0, 0, 0))
        paste_position = (
            (self.target_size - image.size[0]) // 2,
            (self.target_size - image.size[1]) // 2,
        )
        new_im.paste(image, paste_position)
        return new_im

    def _validate_input(self, input_object):
        """
        Ensures the input is a PIL image.

        Args:
            input_object: The input image to validate.

        Returns:
            The input image if it is a PIL image.
        """
        if not isinstance(input_object, Image.Image):
            raise ValueError(f"Expected a PIL image, but got {type(input_object)}")
        return input_object

    def __call__(self, input_object: Image.Image) -> Image.Image:
        """
        Applies resizing and padding transformation

        Args:
            input_object: The input image to resize and pad.

        Returns:
            Image.Image: The resized and padded image.
        """
        validated_image = self._validate_input(input_object)
        self.image_width, self.image_width = validated_image.size

        self._check_dimensions()

        if max(self.image_width, self.image_width) > self.target_size:
            validated_image = self._shrink_image(validated_image)

        return self._add_padding(validated_image)


# Dictionary defining different augmentation levels and their  transformations
additional_transformations = {
    "high": [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomPerspective(distortion_scale=0.5, p=0.2),
        T.RandomPosterize(bits=5, p=0.2),
        T.RandomSolarize(threshold=128, p=0.2),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        T.RandomAutocontrast(p=0.5),
        T.ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.8, 1.2),
            saturation=(0.5, 1.0),
            hue=(-0.01, 0.01),
        ),
        T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.0), shear=(1, 1, 1, 1)
        ),
        T.RandomResizedCrop(size=(224, 224), scale=(0.35, 1.0), ratio=(0.9, 1.1)),
    ],
    "medium": [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomPerspective(distortion_scale=0.5, p=0.5),
        T.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.5, 1.5),
            saturation=(0.5, 1.5),
            hue=(-0.03, 0.03),
        ),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 10.0)),
        T.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(5, 5, 5, 5)
        ),
        T.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), ratio=(0.9, 1.1)),
    ],
    "low": [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=(0, 180)),
    ],
    "random_mix": [
        RandomHorizontalFlip(),
        RandomRotation(180),
        AugMix(),
    ],
}


def define_resize_transformation(
    target_size: Union[tuple[int, int], int] = (224, 224),
    resize_with_proportions: bool = False,
) -> list:
    """
    Defines the resize transformation for the image pipeline.

    Args:
        target_size: The target dimensions for resizing.
        resize_with_proportions: Whether to resize the images while maintaining their proportions.

    Returns:
        list: A list containing the chosen resize transformation.
    """
    return (
        [ResizeWithProportions(target_size)]
        if resize_with_proportions
        else [Resize(target_size)]
    )


def define_transformation_pipeline(
    train: bool = False,
    augmentation_level: str = "low",
    resize_with_proportions: bool = False,
    target_size: Union[tuple[int, int], int] = (224, 224),
    normalize_images: bool = False,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> Compose:
    """
    Defines the transformation pipeline for the image transformations.

    Args:
        train: Flag indicating whether the transformation pipeline is for training or validation/testing. 
                Influences if a augmentation is applied.
        augmentation_level: The level of augmentation to apply to the train images ["low", "medium", "high", "random_mix"].
        resize_with_proportions: Flag indicating whether to resize the images while maintaining their proportions.
        target_size: The target dimensions for the resizing.
        normalize_images: Whether to normalize the images.
        mean: The mean values for normalization. Default are the ImageNet mean values.
        std: The standard deviation values for normalization Default are the ImageNet standard deviation values.

    Returns:
       A transformation pipeline for the images compatible with PyTorch "torchvision.transforms.Compose".
    """

    # Define the base resizing transformation
    resize_transformation = define_resize_transformation(
        target_size=target_size, resize_with_proportions=resize_with_proportions
    )

    # Select transformations based on mode (train or validation)
    if train:
        defined_transformations = resize_transformation + additional_transformations[augmentation_level] + [T.Resize(224), T.ToTensor()]

    else:
        defined_transformations = resize_transformation + [T.Resize(224), T.ToTensor()]

    # Apply normalization if required
    if normalize_images:
        defined_transformations = defined_transformations + [(Normalize(mean, std))]

    return Compose(defined_transformations)


if __name__ == "__main__":
    import pathlib
    import matplotlib.pyplot as plt

    FILE_PATH = r"data/ZooLake2/ZooLake2/ZooLake2.0/aphanizomenon/SPC-EAWAG-0P5X-1570543372901157-3725350526242-001629-055-1224-2176-84-64.jpeg"
    OUTPUT_FOLDER = "./output"

    img_example = Image.open(FILE_PATH)

    if not pathlib.Path(OUTPUT_FOLDER).exists():
        pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        with open(pathlib.Path(OUTPUT_FOLDER, ".gitignore"), "w") as gitignore_file:
            gitignore_file.write("*")

    new_size = (224, 224)

    # plot a example of the resize transformation

    ## preparations
    pipeline_with_porpoptions = define_transformation_pipeline(
        train=False, target_size=new_size, resize_with_proportions=True
    )
    pipeline_without_porpoptions = define_transformation_pipeline(
        train=False, target_size=new_size, resize_with_proportions=False
    )

    tensor_with_porpoptions = pipeline_with_porpoptions(img_example)
    tensor_without_porpoptions = pipeline_without_porpoptions(img_example)

    # show the different images
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))

    ax[0].imshow(img_example)
    ax[0].set_title("Original Image", fontsize=10)

    # change the tensor to numpy array and change the order of the channels
    ax[1].imshow(tensor_with_porpoptions.permute(1, 2, 0))
    ax[1].set_title("Resized with Proportions", fontsize=10)

    ax[2].imshow(tensor_without_porpoptions.permute(1, 2, 0))
    ax[2].set_title("Resized without proportions", fontsize=10)
    plt.suptitle("Resizing with and without proportions", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/resized_image_comparison.jpg")

    # prepare the plot for the different augmentation levels

    torch.manual_seed(1)
    augmentation_levels = ["low", "medium", "high", "random_mix"]

    # create a map function to create the different transformations to include
    # the augmentation level and resize with proportions flag
    def create_pipelines(augmentation_level, resize_with_proportions, image):
        """Mini function to create the different pipelines based on the augmentation level"""
        pipeline = define_transformation_pipeline(
            target_size=new_size,
            resize_with_proportions=resize_with_proportions,
            train=True,
            augmentation_level=augmentation_level,
        )
        return pipeline(image)

    # create a dictionary with all the transformations
    all_augs = {
        augmentation_level: create_pipelines(augmentation_level, True, img_example)
        for augmentation_level in augmentation_levels
    }

    fig, ax = plt.subplots(1, 5, figsize=(10, 4))

    # plot the original image
    ax[0].imshow(img_example)
    ax[0].set_title("Original Image", fontsize=10)
    ax[0].axis("off")

    # plot the different transformations
    for i, (transformation_name, tensor_example) in enumerate(all_augs.items(), 1):
        img = tensor_example.permute(1, 2, 0)

        ax[i].imshow(img)
        ax[i].set_title(f"{transformation_name.replace('_', ' ')}", fontsize=10)
        ax[i].axis("off")

    plt.suptitle("Example of diffrent augementation settings", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/example_augmentations.jpg")
