#!/usr/bin/env bash

# source activate <your_conda_env>

# "data" specifies the dataset name
# "train_path" specifies the training data path
# "val_path" specifies the valid data path
data='facestar_whisper_phi'
python finetune/robust_ger.py --data ${data} \
       --train_path /fs/gamma-projects/audio/facestar_whisper/facestar_full_train_whisper_phi.pt \
       --val_path /fs/gamma-projects/audio/facestar_whisper/facestar_full_test_whisper_phi.pt \
