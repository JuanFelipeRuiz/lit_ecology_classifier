#!/bin/bash
#SBATCH --job-name="test_2epoch"  
#SBATCH --account="em09"
#SBATCH --constraint='gpu'
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --gpus-per-node=1
export OMP_NUM_THREADS=12 #$SLURM_CPUS_PER_TASK
cd ${SCRATCH}/plankton_classifier/lit_ecology_classifier
module purge
module load daint-gpu cray-python
source lit_ecology/bin/activate
python -m lit_ecology_classifier.main --max_epochs 2 --dataset ZooLakeTest --priority config/priority.json  --datapath $HOME/store/empa/em09/aquascope/zoo3.tar --batch_size 128 