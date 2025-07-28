#!/bin/bash
#SBATCH --job-name=rl_train
#SBATCH --output=logs/train_int_%j.log
#SBATCH --time=12:00:00           # 12 horas máximo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # ajusta al nº de cores que quieras
#SBATCH --gres=gpu:1              # si quieres GPU
#SBATCH --mem=64G                 # memoria total

# Carga módulos o activa tu entorno
module load cuda/12.1             # ejemplo, si corres directo en el host
# Si debes iniciar tu contenedor enroot:
# enroot start --name myjob your_image

cd $HOME/tu/proyecto/cuenca-flow-optimizer
export SUMO_HOME=/opt/sumo
export PATH=$SUMO_HOME/bin:$PATH
export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH

# Lanza el entrenamiento
python scripts/train_int.py

