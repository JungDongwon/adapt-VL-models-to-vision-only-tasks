#!/usr/bin/env bash
set -eo pipefail

# Note: This script is assumed to be run from `adaptations/src`

# you need to set the DATA_DIR path, in which the training data is stored
DATA_DIR=./data

# you also need to set the CHECKPOINT_DIR, to which the checkpoints should be saved
CHECKPOINT_DIR=./checkpoints

python finetune_vision_only_task.py \
    --text-dataset $DATA_DIR/adaptations/no-text-features/{train,val}.jsonl \
    --train-image-features-path $DATA_DIR/image-features/tiny-imagenet/train_image_features.hdf5 \
    --valid-image-features-path $DATA_DIR/image-features/tiny-imagenet/valid_image_features.hdf5 \
    --evaluate-every 40 \
    --checkpoint-every 40 \
    --checkpoint-max 1 \
    --checkpoint-dir $CHECKPOINT_DIR/clip-bert-lxmert-middle-lr \
    --model "clip-bert" \
    --bert-checkpoint "./models/model-weights/clipbert/mp_rank_00_model_states.pt" \
    --batch-size 64 \
    --tensorboard-logdir "./logs/no-text-features-clipbert" \
    --lr 0.005
