#!/bin/bash
#SBATCH --job-name=internvl-test
#SBATCH -A umin_kurs_datascismartcity2526
#SBATCH -t 00:20:00
#SBATCH -p scc-gpu
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=minyan.fu@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

# 进入项目目录
cd /user/minyan.fu/u23252/.project/dir.project/minyan/test/test_masterds

# 清空系统环境，防止冲突
module purge

# 加载集群的 Python 和 pip
module load python/3.11.9
module load py-pip/23.1.2
module load py-setuptools/69.2.0
module load py-wheel/0.41.2

# 打印基础信息
echo "Working dir: $(pwd)"
echo "Running on node: ${SLURM_NODELIST}"
which python
python --version
nvidia-smi

# 第一次运行时创建虚拟环境
if [ ! -d "job_venv" ]; then
  echo "Creating venv on compute node..."
  python -m venv job_venv
fi

# 激活虚拟环境
source job_venv/bin/activate

# 离线安装依赖（从 wheels 目录）
echo "Installing dependencies from local wheels..."
pip install --no-index --find-links=wheels -r requirements.txt
pip install --no-index --find-links=wheels timm

# 进入脚本目录并运行
cd scripts
python -u test.py
