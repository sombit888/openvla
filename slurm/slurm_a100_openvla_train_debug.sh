#!/bin/bash
#SBATCH --job-name=full_finetuning_1.0   # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-gpu=12
#SBATCH --nodelist=gcp-eu-2
#SBATCH --gres=gpu:a100-40g:8     # Request 1 A100 GPUs
#SBATCH --time=48:00:00        
#SBATCH --mem-per-gpu=64G
##SBATCH --reservation=sombit-8xa100
#SBATCH --output=/scratch/sombit_dey/job_%j.out
#SBATCH --error=/scratch/sombit_dey/job_%j.err

# bash /home/sombit_dey/.bashrc

export CUDA_HOME=/opt/modules/nvidia-cuda-11.8.0/
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

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
    micromamba run -n gpu_slurm_3.11 pip install flash-attn --no-build-isolation
fi
git lfs install 
cd /scratch/sombit_dey/ && git clone https://huggingface.co/openvla/openvla-7b-prismatic
cd /scratch/sombit_dey/vision_code/openvla/

echo "Running the script"
micromamba run -n gpu_slurm_3.11 torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint /scratch/sombit_dey/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --vla.type prism-dinosiglip-224px+mx-bridge_debug_e6 \
  --data_root_dir /data/work2-gcp-europe-west4-a/nikolay_nikolov/datasets/oxe/resized/ \
  --run_root_dir /scratch/sombit_dey/orig_bridge \
  --run_id orig_bridge \
  --is_resume False 
# micromamba run -n gpu_slurm_3.11 torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  # --pretrained_checkpoint /scratch/sombit_dey/gradual_dino_siglip/gradual_dino_siglip/checkpoints/step-080000-epoch-01-loss=0.4160.pt \
  # --vla.type prism-dinosiglip-224px+mx-bridge_debug_e6 \
  # --data_root_dir /data/work2-gcp-europe-west4-a/nikolay_nikolov/datasets/oxe/resized/ \
  # --run_root_dir /scratch/sombit_dey/gradual_dino_siglip \
  # --run_id gradual_dino_siglip \
  # --is_resume True \
  # --resume_step 80000 \
  # --resume_epoch 01
# micromamba run -n gpu_slurm_3.11 torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
#   --pretrained_checkpoint /scratch/sombit_dey/flip_dino_open_x/flip_dino_open_x/checkpoints/step-447500-epoch-02-loss=0.7219.pt \
#   --vla.type prism-dinosiglip-224px+mx-bridge_debug_e6 \
#   --data_root_dir /data/work2-gcp-europe-west4-a/nikolay_nikolov/datasets/oxe/resized/ \
#   --run_root_dir /scratch/sombit_dey/flip_dino_open_x \
#   --run_id flip_dino_open_x \
#   --is_resume True \
#   --resume_step 447500 \
#   --resume_epoch 2  
  

# micromamba run -n gpu_slurm_3.11 torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
#   --pretrained_checkpoint /scratch/sombit_dey/flip_dino_siglip_fractal/flip_dino_siglip_fractal/checkpoints/step-090000-epoch-01-loss=0.7432.pt \
#   --vla.type prism-dinosiglip-224px+mx-bridge_debug_e6 \
#   --data_root_dir /data/work2-gcp-europe-west4-a/nikolay_nikolov/datasets/oxe/resized/ \
#   --run_root_dir /scratch/sombit_dey/flip_dino_siglip_fractal_new/ \
#   --run_id flip_dino_siglip_fractal_new \
#   --is_resume True \
#   --resume_step 090000 \
#   --resume_epoch 1 \
# micromamba run -n gpu_slurm_3.11 torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
#   --pretrained_checkpoint /work/sombit_dey/flip_dino_siglip_fractal_new/checkpoints/step-105000-epoch-01-loss=0.2683.pt \
#   --vla.type prism-dinosiglip-224px+mx-bridge_debug_e6 \
#   --data_root_dir /data/work2-gcp-europe-west4-a/nikolay_nikolov/datasets/oxe/resized/ \
#   --run_root_dir /scratch/sombit_dey/flip_dino_siglip_fractal_new/ \
#   --run_id flip_dino_siglip_fractal_new \
#   --is_resume True \
#   --resume_step 105000 \
#   --resume_epoch 1 \





# micromamba run -n gpu_slurm_3.11 torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
#   --pretrained_checkpoint /scratch/sombit_dey/model_merging_alpha_0.9/model_merging_alpha_0.9/checkpoints/step-137500-epoch-02-loss=0.5986.pt \
#   --vla.type prism-dinosiglip-224px+mx-bridge_debug_e6 \
#   --data_root_dir /data/work2-gcp-europe-west4-a/nikolay_nikolov/datasets/oxe/resized/ \
#   --run_root_dir /scratch/sombit_dey/model_merging_alpha_1.0/ \
#   --run_id model_merging_alpha_1.0 \
#   --is_resume True \
#   --resume_step 137500 \
#   --resume_epoch 2
#   --vla.type prism-dinosiglip-224px+mx-fractal \
  