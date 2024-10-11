#!/bin/bash
#SBATCH --job-name=full_finetuning    # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-gpu=12
#SBATCH --nodelist=gcp-eu-2
#SBATCH --gres=gpu:a100-40g:8     # Request 1 A100 GPUs
#SBATCH --time=24:00:00        
#SBATCH --me-per-gpu=60G
#SBATCH --output=/scratch/sombit_dey/job_%j.out
#SBATCH --error=/scratch/sombit_dey/job_%j.err

# bash /home/sombit_dey/.bashrc

# export CUDA_HOME=/opt/modules/nvidia-cuda-11.8.0/
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export PATH=$CUDA_HOME/bin:$PATH

# ENV_NAME="simpler_slurm"
ENV_NAME="gpu_slurm_3.11"

# cd /scratch/sombit_dey/SimplerEnv/
rsync -avP /home/sombit_dey/vision_code/openvla/ /scratch/sombit_dey/vision_code/openvla/ --exclude runs --exclude adapter-tmp --exclude wandb

if micromamba env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' exists."
else
    micromamba create -n gpu_slurm_3.11 python=3.11 --yes
    echo "Created conda env gpu_slurm_3.11"
    # micromamba activate gpu_slurm_3.11
    cd /scratch/sombit_dey/vision_code/openvla/
    micromamba run -n gpu_slurm_3.11 pip install -r requirements-min.txt
    micromamba run -n gpu_slurm_3.11 pip install -e . 
    ## install flashattention 
    micromamba run -n gpu_slurm_3.11 pip install flash-attn --no-build-isolation
fi
cd /scratch/sombit_dey/vision_code/openvla/

echo "Running the script"
micromamba run -n gpu_slurm_3.11 torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint /scratch/sombit_dey/vision_code/openvla/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir /data/work2-gcp-europe-west4-a/nikolay_nikolov/datasets/oxe/resized/ \
  --run_root_dir /scratch/sombit_dey/results_full_finetuning/ \
  --is_resume False
  # --vla.type prism-dinosiglip-224px+mx-fractal \
