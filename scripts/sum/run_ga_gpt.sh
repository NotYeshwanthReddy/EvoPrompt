#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  

BUDGET=10
POPSIZE=10
GA=topk

for dataset in sam
do
OUT_PATH=outputs/sum/$dataset/gpt/all/ga/bd${BUDGET}_top${POPSIZE}_para_topk_init/$GA/turbo
mkdir -p $OUT_PATH
for SEED in 5
do
python run.py \
    --seed $SEED \
    --dataset $dataset \
    --task stb \
    --batch-size 10 \
    --prompt-num 0 \
    --sample_num 10 \
    --language_model gpt \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position pre \
    --evo_mode ga \
    --llm_type turbo \
    --setting default \
    --initial all \
    --initial_mode para_topk \
    --ga_mode $GA \
    --cache_path $OUT_PATH/prompts_gpt.json \
    --output $OUT_PATH > $OUT_PATH/log.txt 2>&1
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done
done