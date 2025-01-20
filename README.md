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

The overview.py modul allows the user to create a image overview of one or multiple versions of the given data sets. It calcualtes the hash of the image, to check if they are equal images inside of the given data sets. 

Currently the overview generation is only possible with images from a Dual Scripps Plankton Camera (DSPC), since the image processor generates features from the specific file name. 

#### How to use

To run the overview, it is necessary to create a dictionary containing the version and path to the different image datasets. The version data dictionary can be passed directly to the cmd argument or as a JSON file (e.g. [config/dataset_versions.json](config/dataset_versions.json)).The json fileneeds to contain an abbreviation for the dataset version
and a path to the dataset version folder, suitable for the operation system beeing used.


```bash
 python lit_ecology_classifier/overview.py --dataset Zoo  --image_version_path_dict "config/dataset_versions.json" 
```

#### Args for overview
- `--dataset`: Name of the folder to store non train specific artifacts like image overview. Default: pytho
- `--overview_filename`: Name of the overview file to load/save. Default: overvirew.csv
- `--image_version_path_dict`: Dictionary or path to the json file containing the image versions and their corresponding paths.
- `--summarise_to`: If a path is given, all images of the given versions are summarised into one single folder at given path. Default: None

### Split

```bash
python lit_ecology_classifier/split.py  --priority_classes 'config/priority.json' --rest_classes 'config/rest_classes.json' --dataset "Zoo"
```
    
- `--split_hash`, Hash of the split to reuse. If empty, no hash search is used
- `--split_strategy`",  Split strategy to use. Needs to be one thats implmented in the lit_ecology_classifier/split_strategies folder
- `--filter_strategy`", Filter strategy to use. Needs to be one thats implmented in the lit_ecology_classifier/filter_strategies folder
- `--description`", help ="A optional description of split

# Args for the split process, that can be loaded from a json file
- `--split_args`, Path to the file containing the arguments for the split strategy")
- --filter_args",  Args or path to file containing the arguments for the filter strategy")
- --class_map", type=load_dict, default= {}, help="Args or path to file containing the arguments for the filter strategy")
- --priority_classes", type= load_class_definitions, default=[], help="List of priority classes or path to the JSON file containing the priority classes")
- --rest_classes", type=load_class_definitions, default=[], help="List of rest classes or path to the JSON file containing the rest classes")

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
