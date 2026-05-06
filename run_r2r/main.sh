#!/bin/bash

flag=" --exp_name exp_1
      --run-type eval
      --exp-config vlnce_baselines/config/exp2.yaml
      --nprocesses 8
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0,1] 
      SIMULATOR_GPU_IDS [0,1]
      "
CUDA_VISIBLE_DEVICES=0,1 python run_mp.py $flag 
