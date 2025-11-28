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

# 加载集群的 Python 和 pip
module python/3.11.9

python --version || exit 1

if [ ! -d "job_venv" ]; then
  echo "Creating venv..."
  python -m venv job_venv
fi

source job_venv/bin/activate

echo "Installing deps from wheels..."
python -m pip install --no-index --find-links=wheels -r requirements.txt

cd scripts
python -u test.py
