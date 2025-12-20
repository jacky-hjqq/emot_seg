#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --nodelist=a0932
#SBATCH --gres=gpu:a100:2
#SBATCH --time=23:59:00
#SBATCH --job-name=pose_sam6d
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=NONE

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
 
unset SLURM_EXPORT_ENV
module load python
conda activate vggt_pose_sam6d
# install pointnet2
pip uninstall -y pointnet2
cd ~/projects/vggt_pose/vggt/sam6d/model/pointnet2
rm -rf build
rm -rf pointnet2.egg-info
TORCH_CUDA_ARCH_LIST="8.0" pip install .

mkdir -p /home/atuin/v120bb/v120bb15/Omni6DPoseAPI/data/Omni6DPose
squashfuse /home/atuin/v120bb/v120bb15/Omni6DPoseAPI/data/Omni6DPose.sqsh /home/atuin/v120bb/v120bb15/Omni6DPoseAPI/data/Omni6DPose

cd $TMPDIR
mkdir YCBV
tar xf $WORK/ycbv/ycbv_pbr.tar -C YCBV

cd ~/projects/vggt_pose
python train_sam6d.py 