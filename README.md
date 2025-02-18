# Lit Ecology Classifier
Documentation: https://lit-ecology-classifier.readthedocs.io/en/latest/

Lit Ecology Classifier is a machine learning project designed for image classification tasks. It leverages PyTorch Lightning for streamlined training and evaluation processes.

**Features:**

- Easy configuration and setup
- Utilizes PyTorch Lightning for robust training and evaluation
- Supports training on multiple GPUs
- Test Time Augmentation (TTA) for enhanced evaluation
- Integration with Weights and Biases (WandB) for experiment tracking

## Set up

The Lit Ecology Classifier is designed to be easy to set up and use. The following instructions will guide you through the process of
setting up the project and running a example with the provided datasets. 


### Environment

To set up the Lit Ecology Classifier, make sure to have Python installed in your environment. Afterwards, install the package.

```bash
pip install git+https://github.com/JuanFelipeRuiz/lit_ecology_classifier    
```

### Data 

For an hand on experience or reporoduction of research results, the Lit Ecology Classifier provides the built in module `prepare_eco_data` to download and
prepare the datasets. The supported datasets are: 


| Dataset        | Description                       |
| :-----------:    | :-------|
| mini_dataset   |  Subversion of ZooLake2 dataset for the Plant Science Symposium *Machine Learning in Plant and Environmental Sciences*. 
| ZooLake1      |  Initial plankton dataset with a tota of 17'900 labelled images. Further information can be found in the papers ["Data for: Underwater dual-magnification imaging for automated lake plankton monitoring"](https://opendata.eawag.ch/dataset/data-for-underwater-dual-magnification-imaging-for-automated-lake-plankton-monitoring) by Merz, E. et al. and ["Deep Learning Classification of Lake Zooplankton"]((https://opendata.eawag.ch/dataset/deep-learning-classification-of-zooplankton-from-lakes)) from Kyathanahally, S et al. (2021)     |
| ZooLake2      | Second version of the planhkton data set wich include a total of 24'000 annotated images and the introduction of the *out-of-dataset (OOD)*. The OOD was utilised by C. Cheng et al. (2024) in their research into ["Producing plankton classifiers that are robust to dataset shift"](https://data.eawag.ch/dataset/data-for-producing-plankton-classifiers-that-are-robust-to-dataset-shift).|
| OOD           | "Out of Dataset", images used to measure the dataset shift inside the reseacht of C.Cheng et al. (2024) in ["Producing plankton classifiers that are robust to dataset shift"](https://data.eawag.ch/dataset/data-for-producing-plankton-classifiers-that-are-robust-to-dataset-shift).| 
| ZooLake3      | comming soon.... |
| Pytholake1    | comming soon.... |


#### Dowloand
With the `prepare_eco_data` module, the user can download the datasets. The modul will download the dataset and extract it into the `data` folder. If no `data` folder exists, it will be created in the root of the project. 

The modul will also create a `dataset_versions.json` file in the `config` folder with the path to the downloaded dataset for further use. If a dataset_versions.json already exists and may differm the user will be asked how to interact with it.

To select the dataset to download, use the `--dataset` argument with the exact dataset name from the table above.

``` bash & cmd
python -m lit_ecology_classifier.prepare_eco_data --dataset mini_dataset
```

The user may be ask one of the following questions:
 
 - **Exisiting dataset.zip found**

    When a zip file with the same name as the dataset is already present in the `data folder`. Choose *`y`* to extract the zip file or *`n`* to keep the existing zip file and continue with the download. 

    >***Winter School Participants***
    > 
    > The mini ood and dataset used for the workshop are currently available as ZIP file inside the data folder. Please ensure the ZIP file is present before running the prepare eco module and press *`y`* when asked per cmd.
    >  
 
 - ***Overwrite?*** 

    Case that appears if a folder with the name of the dataset is already present in the `data folder`. Choose *`y`* to overwrite the folder or *`n`* to keep the existing folder and cancel the download. A download of the dataset is not possible if a folder with the same name is already present and should not be overwritten.

 - ***Existing config file 'dataset_versions.json'found:***

     To simplify the setup, the script may ask the user if the dataset should be added inside of `dataset_versions.json` or overwrite it. 
      file if it is not already inside. ***Winter School Participants** should chose overwrite[1], for the workshop* 


#### Manual download

The download can be made manually from the links in the description. Please follow following steps:

1) Download the zip file from the provided link in the description. 
2) Move the zip file to the `data` folder in the root of the project.
3) Ensure the zip file is named as the dataset. Eg. the donwload of `ZooLake1` is as default `data` and should be renamed to `ZooLake1.zip`.
4) Run the `prepare_eco_data` module with the `--dataset` argument to extract the zip file. Example:

    ``` bash & cmd
    python -m lit_ecology_classifier.prepare_eco_data --dataset ZooLake1
    ```




## Examples

This section provides scripts examples for the different module to use. For more hands on examples, you may prefer to use the provided Jupyter notebooks in the `notebooks` folder. Example of the main modules and there corresponding notebooks are listed below:


| Module        | Notebooks              |Description                    |
| ------------    | --------|--------|
| [`overview.py`](#Overviewpy)  |  | Create an image overview of the given data sets. |
| [`split.py`](#Split)  |  | Create a new split of the given data set. |
| [`main.py`](#training)  |  | Train a model on the given data set. |
| [`predict.py`](#inference)  |  | Run inference on unlabelled data. |



### Overview.py

The overview.py file allows the user to create a image overview of one or multiple versions of the given data sets. It calcualtes the hash of the image, to check if they are idetical images inside of the given data sets. 

Currently the overview generation is only possible with images from a Dual Scripps Plankton Camera (DSPC), since the image processor generates features from the specific generated file name. 


#### Additional set up to run overview.py

Necessarily: 
- `dataset_version_path_dict`:


    A json file (e.g. [config/dataset_versions.json](config/dataset_versions.json)) containing the version and path to the different image datasets, automatically generated by the `get_data.py` script. The json file needs to contain an abbreviation for the dataset version and a working path to the dataset version folder. The abbreviation is used to reference the dataset version in the overview table and are used for further processing of the data. 

    ```json
    {"MiniZoolake": "data/MiniZoolake"}
    ```

#### Run the overview module

```bash
python -m lit_ecology_classifier.overview --dataset Zoo  --dataset_version_path_dict "config/dataset_versions.json" 
```

#### Args for overview

necessarily arguments: 
- `--dataset`: Name of the folder to store non train specific artifacts like image overview.
- `--dataset_version_path_dict`: Path to the config json containing the image versions and their corresponding paths.

Use case specific arguments:

- `--summarise_to`: If a path is given, all images of the given versions are summarised and copied into one single folder at the given path. **Needed to train a model based on multiple versions**.


Optional arguments: 
- `--overview_filename`: Name of the overview file to load/save. Default: overview.csv

### Split

`Split` is a modul that allows the user to reload and create new splits. The used arguments and description are stored in a split.overview file inside the dataset folder. The data of each split is stored in a  dataframes inside of the given `args.dataset/split` containing the split overview and a split folder containg the splits.  

The usage of own filter or split strategies are not possible with a direct use of the split module. To use own
splits, create a own split module.

#### Additional set up:

- `overview.csv`:
    Image overview inside the `arg.dataset` folder. Can be generated with [overview.py](#overviewpy).

- `priority_classes.json`:
    List of classes to prioritize. Sets all other classes/labels to "rest : 0 " (e.g. [config/priority_classes.json](config/priority_classes.json). The priority_classes.json needs to contain the key `priority_classes` and the priority list as value to work properly.

- `rest_classes.json`:
    List of classes to keep alongside the priority classes (e.g. [config/rest_classes.json](config/rest_classes.json). All classes/labels not defined inside prio or rest classes are removed.The rest_classes.json need to contain the key `rest_classes` ant the list as value 


#### Run the split module
```bash
python -m lit_ecology_classifier.split  --priority_classes 'config/priority_classes.json' --rest_classes 'config/rest_classes.json' --dataset "Zoo"
```
#### Args for the split module

Following argument is necessarily: 


- `--dataset`: Name of the folder to store non train specific artifacts like image overview. 

Use case specific arguments:

- `--split_hash`:  Hash of the split to reload.
- `--overview_filename`: Name of the overview file to load/save.
- `--split_strategy`: Split strategy to use. Needs to be a strategy that is implemented in the lit_ecology_classifier/split_strategies folder.
- `--filter_strategy`: Filter strategy to use. Needs to be one that is implemented in the lit_ecology_classifier/filter_strategies folder.
- `--description`: Description for of the split for the split overview.
- `--split_args`: Path to json containing the args to use for the split strategy in a dictionary format.
- `--filter_args`: Path to json containing the args to use for the  filter strategy in a dictionary format.
- `--class_map`: Class map dictionary or path to json containing the class mapping
- `--priority_classes`: List of defined priority classes or path to JSON file containing the priority classes in a {"priority_classes" : [class1,class2]} format.
- `--rest_classes`: List of defined rest classes or path to the JSON file containing the rest classes.

### Training

To only train a model, the train modul can be used. It is independent of the split and overview moduls. It uses a random split with the given split ratio with the given values.

#### Additional Setup 

Necessarily set up:

- `get_model` a model backbone of beitv2 can be downloaded using the get_model.sh script.

    ```bash
    source get_model.sh
    ```

use case specific set up: 
- `priority_classes.json`:
    List of classes to prioritize. Sets all other classes/labels to "rest : 0 " (e.g. [config/priority_classes.json](config/priority_classes.json). The priority_classes.json needs to contain the key `priority_classes` and the priority list as value to work properly.

- `rest_classes.json`:
    List of classes to keep alongside the priority classes (e.g. [config/rest_classes.json](config/rest_classes.json). All classes/labels not defined inside prio or rest classes are removed.The rest_classes.json need to contain the key `rest_classes` ant the list as value 

- `wandb`: The lit_eco_classifier supports the tracking of the training and experiments. To use follow the [wandb quickstart](https://docs.wandb.ai/quickstart/) until step 2 
#### Run train:  

```bash
python -m lit_ecology_classifier.main --max_epochs 2 --dataset phyto --priority config/priority.json --datapath data/ZooLake2
```

### Training Arguments

- `--datapath`: Path to the tar file containing the training data. Can be a tar or image folder
- `--train_outpath`: Output path for training artifacts.
- `--dataset`: Path for non training artifacts and name for wand project.
- `--main_param_path`: Main directory where the training parameters are saved.
- `--use_wandb`: Use Weights and Biases for logging.
- `--priority_classes`: Path to the JSON file with priority classes.
- `--balance_classes`: Balance the classes for training.
- `--batch_size`: Batch size for training.
- `--max_epochs`: Number of epochs to train.
- `--lr`: Learning rate for training.
- `--lr_factor`: Learning rate factor for training of full body.
- `--no_gpu`: Use no GPU for training.

### Inference

To run inference on unlabelled data, use the following command:

```bash
python -m lit_ecology_classifier.predict --datapath ZooLake2/Predict --model_path phyto_priority_cyanos.ckpt --outpath ./predictions/
```
### Inference Arguments

- `--outpath`: Directory where predictions are saved.
- `--model_path`: Path to the model file.
- `--datapath`: Path to the tar file containing the data to classify.
- `--no_gpu`: Use no GPU for inference.
- `--no_TTA`: Disable test-time augmentation.


## Documentation

Detailed documentation for this project is available at [Read the Docs](https://lit-ecology-classifier.readthedocs.io).

