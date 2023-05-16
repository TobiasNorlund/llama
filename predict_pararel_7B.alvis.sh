#!/usr/bin/env bash
#SBATCH --gpus-per-node A100:1
#SBATCH --time 1-00:00:00

module load Anaconda3/2021.05
source /cephyr/users/tobiasno/Alvis/.bashrc
conda activate /cephyr/users/tobiasno/Alvis/denitsa-shared/envs/llama

relations=("P937" "P1412" "P127" "P103" "P276" "P159" "P140" "P136" "P495" "P17" "P361" "P36" "P740" "P264" "P407" "P138" "P30" "P131" "P176" "P449" "P279" "P19" "P101" "P364" "P106" "P1376" "P178" "P413" "P27" "P20")

set -x
for relation in "${relations[@]}"
do
    torchrun --nproc_per_node 1 predict_pararel.py \
        --model-dir /cephyr/users/tobiasno/Alvis/denitsa-shared/LLaMA/7B \
        --tokenizer-path /cephyr/users/tobiasno/Alvis/denitsa-shared/LLaMA/tokenizer.model  \
        --pararel-examples data/all_n1_atlas/${relation}.jsonl \
        --pararel-options data/all_n1_atlas/${relation}_options.txt \
        --output-json data/pararel_predictions/7B/${relation}.jsonl \
        --output-preds data/pararel_predictions/7B/${relation}.preds.npy \
        --batch-size 64
done