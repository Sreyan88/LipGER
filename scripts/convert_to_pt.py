import os
os.environ['TRANSFORMERS_CACHE'] = '/fs/nexus-projects/brain_project/eccv/RobustGER/cache/hub'
import tiktoken
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, GenerationConfig
from torch.utils.data import DataLoader
from huggingface_hub.hf_api import HfFolder
import datasets
import json

import pandas as pd
import re
from datasets import load_dataset, Dataset

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

from tqdm import tqdm

model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(torch.device("cuda"))

# model_name = "meta-llama/Llama-2-13b-chat-hf"
model_name = "microsoft/phi-2"

import random

def random_sample_sequence(lst, sample_size):
    indices = list(range(len(lst)))
    selected_indices = random.sample(indices, sample_size)
    selected_indices.sort()
    sampled_list = [lst[i] for i in selected_indices]
    return sampled_list

def calculate_noise_embedding(n_best_hypotheses):
    embeddings = model.encode(n_best_hypotheses)
    n = len(n_best_hypotheses)
    embedding_diffs = []
    for i in range(n):
        for j in range(i):
            diff = embeddings[i] - embeddings[j]
            embedding_diffs.append(diff)
    noise_embedding = np.stack(embedding_diffs, axis=0)
    return noise_embedding


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, access_token="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", padding_side='left')

prompt_1 = 'Below is the best-hypotheses transcribed from speech recognition system. Please try to revise it using the words which are only included into other-hypothesis, and write the response for the true transcription.\n\n### Best-hypothesis:\n'
prompt_2 = '\n\n### Other-hypothesis:'
prompt_3 = '\n\n### Response:\n'

all_pt = []

with open('/fs/gamma-projects/audio/EasyComDataset_Processed/easycom_test_avsr.json', 'r') as f:
    all_files = json.load(f)

for item in tqdm(all_files):

    new_dict = {'id': item['Uid']}

    item["nhyps_base"] = [i.lower() for i in item["nhyps_base"]]
    item["Caption"] = item["Caption"].lower()

    other_hypothesis = random_sample_sequence(item["nhyps_base"][1:], 5)
    final_prompt_no_response = prompt_1 + item["nhyps_base"][0] + prompt_2 + '\n' + ('\n').join(other_hypothesis) + prompt_3
    final_prompt = final_prompt_no_response + item["Caption"] + '<|endoftext|>' #'<|endoftext|>' #'</s>'

    input_ids_no_response = tokenizer.encode(final_prompt_no_response)
    input_ids = tokenizer.encode(final_prompt)

    labels_temp = input_ids.copy()
    labels = [-1] * len(input_ids_no_response) + labels_temp[len(input_ids_no_response):]

    new_dict['input_ids_no_response'] = torch.tensor(input_ids_no_response,dtype=torch.int32)
    new_dict['input_ids'] = torch.tensor(input_ids,dtype=torch.int32)
    new_dict['labels'] = torch.tensor(labels,dtype=torch.int32)

    new_dict['input'] = final_prompt
    new_dict['ground_truth'] = item["Caption"]

    noise_embedding = calculate_noise_embedding(other_hypothesis)
    new_dict['emb_diff'] = torch.tensor(noise_embedding)

    new_dict['clean_speech'] = item["whisper_emb"]
    new_dict['noisy_speech'] = item["whisper_emb"]

    new_dict['mouthroi'] = item["Mouthroi"]

    all_pt.append(new_dict)


torch.save(all_pt, 'data/easycom_test_avsr_phi.pt')




    










    






