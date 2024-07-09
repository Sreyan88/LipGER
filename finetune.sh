#!/usr/bin/env bash

# source activate <your_conda_env>

# "data" specifies the dataset name
# "train_path" specifies the training data path
# "val_path" specifies the valid data path

data='facestar_whisper_phi'
python finetune/lipger.py --data ${data} \
       --train_path /absolute/path/to_your_train.pt \
       --val_path /absolute/path/to_your_val.pt \
