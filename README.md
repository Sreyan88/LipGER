# LipGER
Code for InterSpeech 2024 Paper: LipGER: Visually-Conditioned Generative Error Correction for Robust Automatic Speech Recognition


## Training the LipGER model  

### 1. Prepare the checkpoint  

First prepare the checkpoints using:

```
pip install huggingface_hub

python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf --token your_hf_token

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
```

To see all available checkpoints, run:

```
python scripts/download.py | grep Llama-2
```

For more details, you can also refer to [this link](https://github.com/YUCHEN005/RobustGER/tree/master/tutorials)  

### 2. Prepare the lip movements  

### 3. Fine-tune LipGER  

Then just run:

```
sh finetune.sh
```

### LipGER inference  

For inference just run:  

```
infer.sh
```

 