#!/bin/bash
# You need to modify this path
DATASET_DIR="/home/ccyoung/DCase/data"

# You need to modify this path as your workspace
WORKSPACE="/home/ccyoung/Downloads/dcase2018_task1-master"

DEV_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-development"
#DEV_SUBTASK_B_DIR="TUT-urban-acoustic-scenes-2018-mobile-development"
LB_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-leaderboard"
#LB_SUBTASK_B_DIR="TUT-urban-acoustic-scenes-2018-mobile-leaderboard"
#EVAL_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-evaluation"
#EVAL_SUBTASK_B_DIR="TUT-urban-acoustic-scenes-2018-mobile-evaluation"

BACKEND="keras"	# "pytorch" | "keras"
HOLDOUT_FOLD=1
GPU_ID=0

############ Extract features ############
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_A_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_B_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_A_DIR --data_type=evaluation --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_B_DIR --data_type=evaluation --workspace=$WORKSPACE

############ Development subtask A ############
# Train model for subtask A
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py train --model=attention2 --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD --cuda
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py train --a=300 --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD --cuda

# Evaluate subtask A
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=11500 --cuda
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_data_to_truncation  --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=9600 --cuda
#

############ Full train subtask A ############
# Train on full development data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --cuda

# Inference evaluation data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_evaluation_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_A_DIR --eval_subdir=$EVAL_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration=5000 --cuda