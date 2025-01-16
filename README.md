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
bash source get_data.sh ZooLake1
```
Windows:
```powershell
.\get_model.bat ZooLake1
````

## Usage

The lit_ecology_classifer package provides following functionalites/moduls for the the process and classification of the images,

- [`overview.py`](#Overview.py)
- [`split.py`](#Split)
- [`main.py`](#training)
- [`predict.py`](#inference)

### Overview.py

The overview.py modul allows the user to create a image overview of one or multiple versions of the given data sets. It calcualtes the hash of the image, to check if they are equal images inside of the given data sets. 

Currently the overview generation is only possible with images from a Dual Scripps Plankton Camera (DSPC), since the image processor generates features from the specific file name. 

#### How to use

To run the overview, it is necessary to create a dictionary containing the version and path to the different image datasets. The version data dictionary can be passed directly to the cmd argument or as a JSON file (e.g. [config/dataset_versions.json](config/dataset_versions.json).The json fileneeds to contain an abbreviation for the dataset version
and a path to the dataset version folder, suitable for the operation system beeing used.


```bash
 python overview.py --name Zoo  --image_version_path_dict "config/dataset_versions.json" --output "output" 
```
#### Args for overview
- parser.add_argument("--dataset", default="phyto", help="Name of the dataset to store non train specific artifacts") 
- parser.add_argument("--overview_filename", default="overview.csv", help="Name of the overview file to load/save")
- parser.add_argument("--image_version_path_dict", type=load_dict, help="Dictionary or path to the json file containing the image versions and their corresponding paths")
- parser.add_argument("--summarise_to", type= str, default = None , help="If a path is given, the given versions are summarised int to the given path. If empty, no summarisation is done")
### Split


### Training

To train the model, a model backbone of beitv2 can be downloaded using the get_model.sh script.

```bash
source get_model.sh
```

Afterwards, the model can be trained either 

```bash
python -m lit_ecology_classifier.main --max_epochs 20 --dataset phyto --priority config/priority.json
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
