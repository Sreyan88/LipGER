#!/usr/bin/env bash

# "test_data" specifies the test set, e.g., test_chime4/test_real, test_chime4/test_simu

python inference/robust_ger.py --test_data /fs/gamma-projects/audio/EasyComDataset_Processed/easycom_test_avsr_phi.pt
