# LipGER
Code for InterSpeech 2024 Paper: LipGER: Visually-Conditioned Generative Error Correction for Robust Automatic Speech Recognition


## Prepare the environment  

```
pip install -r requirements.txt
```

## Training the LipGER model  

### 1. Prepare the LLM checkpoint  

First prepare the checkpoints using:

```shell
pip install huggingface_hub

python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf --token your_hf_token

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
```

To see all available checkpoints, run:

```shell
python scripts/download.py | grep Llama-2
```

For more details, you can also refer to [this link](https://github.com/YUCHEN005/RobustGER/tree/master/tutorials), where you can also prepare other checkpoints for other models. Specifically, we use [TinyLlama](https://github.com/jzhang38/TinyLlama) for our experiments.  

### 2. Download the checkpoint for the Lip Motion Encoder  

The checkpoint is available [here](https://drive.google.com/file/d/1RqGd_13BeX1ybD1UPJIu8eDsilCsrAF9/view?usp=sharing). After downloading, change the path of the checkpoint at <>.

### 2. Prepare the dataset for training (and inference)  

### 3. Fine-tune LipGER  

To finetune LipGER, just run:

```
sh finetune.sh
```
where you need to manually set the values for `data`, `--train_path` and `--val_path`.

## LipGER inference  

For inference just run:  

```
infer.sh
```

## Acknowledgement  

Our code is inspired from [RobustGER](https://github.com/YUCHEN005/RobustGER/tree/master). Please cite their paper too if you find our paper or code useful.  


## Citation    

```
@inproceedings{
    ghosh2024abex,
    title={{ABEX}: Data Augmentation for Low-Resource {NLU} via Expanding Abstract Descriptions},
    author={Sreyan Ghosh and Utkarsh Tyagi and Sonal Kumar and Chandra Kiran Reddy Evuru and and Ramaneswaran S and S Sakshi and Dinesh Manocha},
    booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
    year={2024},
}
```