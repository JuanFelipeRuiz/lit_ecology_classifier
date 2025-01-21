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

### Enviroment

To set up the Lit Ecology Classifier, make sure to have Python installed in your enviroment. Afterwards, install the package.

```bash
pip install lit-ecology-classifier
```

### Data

To download the data use `get_data.sh`/`get_data.bat` with a supported argument. Supported arguments are:

| Argument        | Description                       |
| ------------    | --------|
| MiniDataset   |  Subversion of ZooLake2 dataset for the Plant Science Symposium *Machine Learning in Plant and Environmental Sciences*. 
| ZooLake1      |  Initial plankton dataset with a tota of 17'900 labelled images. Further information can be found in the papers ["Data for: Underwater dual-magnification imaging for automated lake plankton monitoring"](https://opendata.eawag.ch/dataset/data-for-underwater-dual-magnification-imaging-for-automated-lake-plankton-monitoring) by Merz, E. et al. and ["Deep Learning Classification of Lake Zooplankton"]((https://opendata.eawag.ch/dataset/deep-learning-classification-of-zooplankton-from-lakes)) from Kyathanahally, S et al. (2021)     |
| ZooLake2      | Second version of the planhkton data set wich include a total of 24'000 annotated images and the introduction of the *out-of-dataset (OOD)*. The OOD was utilised by C. Cheng et al. (2024) in their research into ["Producing plankton classifiers that are robust to dataset shift"](https://data.eawag.ch/dataset/data-for-producing-plankton-classifiers-that-are-robust-to-dataset-shift).|
| ZooLake3      | ...  |
| Pytholake1    | ...  |

 Make sure to execute the code within the working folder since the default dowload path is the current working directory.

Linux/Mac:
```bash 
bash get_data.sh ZooLake1
```
Windows:
```powershell
.\get_data.bat ZooLake1
````

## Usage

The lit_ecology_classifer package provides following functionalites/moduls for the the process and classification of the images,

- [`overview.py`](#Overviewpy)
- [`split.py`](#Split)
- [`main.py`](#training)
- [`predict.py`](#inference)

### Overview.py

The overview.py file allows the user to create a image overview of one or multiple versions of the given data sets. It calcualtes the hash of the image, to check if they are equal images inside of the given data sets. 

Currently the overview generation is only possible with images from a Dual Scripps Plankton Camera (DSPC), since the image processor generates features from the specific file name. 

#### Additional set up: 

- `dataset_version_path_dict`:
    A dictionary or .json file containing the version and path to the different image datasets. (e.g. [config/dataset_versions.json](config/dataset_versions.json))
    The .json file needs to contain an abbreviation for the dataset version and a path to the dataset version folder, suitable for the operation system beeing used.

#### Args for overview.py

Necesserly: 
- `--dataset`: Name of the folder to store non train specific artifacts like image overview.
- `--dataset_version_path_dict`: Dictionary or path to the json file containing the image versions and their corresponding paths.

Optional : 
- `--overview_filename`: Name of the overview file to load/save. Default: overvirew.csv
- `--summarise_to`: If a path is given, all images of the given versions are summarised and copied into one single folder at given path. Needed to train a model based on multiple versions.


#### Run
```bash
python lit_ecology_classifier/overview.py --dataset Zoo  --dataset_version_path_dict "config/dataset_versions.json" 
```

### Split.py

`Split.py` is a modul that allows the user to reload and create new splits. The used arguments and description are stored in a split.overview file inside the dataset folder. The data of each split is stored in a  dataframes inside of `args.dataset/split` containing the image and split name.  

The usage of own filter or split stragies are not possbile with the split.py file. 
#### Additional set up:

- `overview.csv`:
    Image overview inside the `arg.dataset` folder. Can be generated with [overview.py](#overviewpy).

- `priority_classes.json`:
    List of classes to priority. Sets all other classes to "rest" (e.g. [config/priority_classes.json](config/priority_classes.json). Can be given as list into the cmd or as path to the priority_classes.json containg the key `priority_classes` ant the list as value. 

- `rest_classes.json`:
    List of classes to keep alongside the  (e.g. [config/rest_classes.json](config/rest_classes.json)). Can be given as list into the cmd or as path to the rest_classes.json containg the key `rest_classes` ant the list as value 


#### 
```bash
python lit_ecology_classifier/split.py  --priority_classes 'config/priority.json' --rest_classes 'config/rest_classes.json' --dataset "Zoo"
```
#### Args for split.py

Following argument is necesserly: 


- `--dataset`: Name of the folder to store non train specific artifacts like image overview. 

Use case specific args:
- `--split_hash`:  Hash of the split to reload.
- `--overview_filename`: Name of the overview file to load/save.
- `--split_strategy`: Split strategy to use. Needs to be a strategy thats implmented in the lit_ecology_classifier/split_strategies folder.
- `--filter_strategy`: Filter strategy to use. Needs to be one thats implmented in the lit_ecology_classifier/filter_strategies folder.
- `--description`: Description for of the split for the split overview.
- `--split_args`:  Arguments in dictionry format or path to json containing the args to use for the split strategy.
- `--filter_args`: Arguments in dictionry format or path to json containing the args to use for the  filter strategy.
- `--class_map`: Class map dictionary or path to json containing the class mapping
- `--priority_classes`: List of defined priority classes or path to JSON file containing the priority classes in a {"priority_classes" : [class1,class2]} format.
- `--rest_classes`: List of defined rest classes or path to the JSON file containing the rest classes.

### Training


To train the model, a model backbone of beitv2 can be downloaded using the get_model.sh script.

```bash
source get_model.sh
```

Afterwards, the model can be trained either 

```bash
python -m lit_ecology_classifier.main --max_epochs 2 --dataset phyto --priority config/priority.json --datapath data/ZooLake2
```

### Inference

To run inference on unlabelled data, use the following command:

```bash
python -m lit_ecology_classifier.predict --datapath ZooLake2/Predict --model_path phyto_priority_cyanos.ckpt --outpath ./predictions/
```

## Configuration

The project uses an argument parser for configuration. Here are some of the key arguments:

### Training Arguments

- `--datapath`: Path to the tar file containing the training data.
- `--train_outpath`: Output path for training artifacts.
- `--main_param_path`: Main directory where the training parameters are saved.
- `--dataset`: Name of the dataset.
- `--use_wandb`: Use Weights and Biases for logging.
- `--priority_classes`: Path to the JSON file with priority classes.
- `--balance_classes`: Balance the classes for training.
- `--batch_size`: Batch size for training.
- `--max_epochs`: Number of epochs to train.
- `--lr`: Learning rate for training.
- `--lr_factor`: Learning rate factor for training of full body.
- `--no_gpu`: Use no GPU for training.

### Inference Arguments

- `--outpath`: Directory where predictions are saved.
- `--model_path`: Path to the model file.
- `--datapath`: Path to the tar file containing the data to classify.
- `--no_gpu`: Use no GPU for inference.
- `--no_TTA`: Disable test-time augmentation.

## Documentation

Detailed documentation for this project is available at [Read the Docs](https://lit-ecology-classifier.readthedocs.io).

### Example SLURM Job Submission Script

Here is an example SLURM job submission script for training on multiple GPUs:

```bash
#!/bin/bash
#SBATCH --account="em09"
#SBATCH --constraint='gpu'
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
export OMP_NUM_THREADS=12 #$SLURM_CPUS_PER_TASK
cd ${SCRATCH}/lit_ecology_classifier
module purge
module load daint-gpu cray-python
source lit_ecology/bin/activate
python -m lit_ecology_classifier.main --max_epochs 2 --dataset phyto --priority config/priority.json
```
