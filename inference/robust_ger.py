import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.robust_ger import generate
from lipger import Tokenizer
from lipger.robust_ger import GPT, Block, Config
from lipger.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization
from evaluate import load
import h5py
from finetune.utils import Compose, Normalize, RandomCrop, HorizontalFlip, CenterCrop

def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([
                                Normalize( 0.0,255.0 ),
                                RandomCrop(crop_size),
                                # HorizontalFlip(0.5),
                                Normalize(mean, std) ])
    preprocessing['val'] = Compose([
                                Normalize( 0.0,255.0 ),
                                CenterCrop(crop_size),
                                Normalize(mean, std) ])
    preprocessing['test'] = preprocessing['val']
    return preprocessing

def load_mouthroi(filename):
    with h5py.File(filename, 'r') as hf:
        return hf['video_frames'][:]

wer = load("wer")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=1, help='lNo of GPUs (default: 1)')
parser.add_argument('--test_data', type=str)
args = parser.parse_args()

devices = args.d

exp_path = f'/path/where/you/want_to_save/'
predict_dir = f'{exp_path}/predictions_avsr_{args.test_data}'  # place to save predictions

data_path = f'{args.test_data}'

precision = None
quantize = None
strategy: str = "auto"
torch.set_float32_matmul_precision("high")

precision = precision or get_default_supported_precision(training=False)
fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
fabric.launch()

dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

checkpoint_dir = Path("/path/to/your/chckpoint/folder") # path to the folder where you prepared your checkpoints
check_valid_checkpoint_dir(checkpoint_dir)

with open(checkpoint_dir / "lit_config.json") as fp:
    config = Config(**json.load(fp))

checkpoint_path = '/path.to/your/adapter.pth'

with fabric.init_module(empty_init=False):
    model = GPT(config)

tokenizer = Tokenizer(checkpoint_dir)
data = torch.load(data_path)

def result(adapter_path, model):
    # LOADING CORRESPOINDG ADAPTER MODEL
    with lazy_load(checkpoint_path) as checkpoint:
        x,y = model.load_state_dict(checkpoint['model'], strict=False)
        print(x)
        print(y)

    model.eval()
    model = fabric.setup(model)

    c = 0
    return_dict = {}
    pr = []
    gt = []
    to_json = []
    for datapoint in data:

        lipreading_preprocessing_func = get_preprocessing_pipelines()['test']

        encoded = datapoint['input_ids_no_response'].to(model.device)
        emb_diff = datapoint['emb_diff'].to(model.device).to(dtype)
        mouth_features = torch.FloatTensor(lipreading_preprocessing_func(torch.from_numpy(load_mouthroi(datapoint['mouthroi'])))).unsqueeze(0).unsqueeze(0).to(model.device).to(dtype)

        ground_truth = datapoint['ground_truth']

        max_returned_tokens = encoded.size(0) + 150

        y = generate(
            model=model,
            emb_diff=emb_diff,
            visual_features=mouth_features,
            idx=encoded,
            max_returned_tokens=max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=0.2,
            top_k=1,
            eos_id=tokenizer.eos_id
        )

        model.reset_cache()
        output = tokenizer.decode(y)

        inf = output[len(tokenizer.decode(encoded)):].split('\n')[0].strip()
        ref = ground_truth.strip()
        if inf == ref:
            c = c + 1
        pr.append(inf)
        gt.append(ref)
        print(ref)
        print(inf)
        to_json.append({'inference': inf, 'ground_truth': ref})

    print(f'For {adapter_path}')
    return_dict['adapter_path'] = adapter_path
    wer_ = wer.compute(predictions=pr, references=gt)
    print(f'WER is {wer_}')
    return_dict['WER'] = wer_
    print(f'Ground truth matches is {c}/{len(data)}')
    to_json.append({'wer': wer_, 'gtms': f'{c}/{len(data)}'})
    return_dict['gtms'] = c / len(data)
    os.system(f'mkdir -p {predict_dir}')
    with open(os.path.join(predict_dir, adapter_path.split('/')[-1].split('.pth')[0] + '.json'), 'w') as f:
        f.write(json.dumps(to_json, indent=4, ensure_ascii=False))
    print(os.path.join(predict_dir, adapter_path.split('/')[-2] + '.json'))
    print('the post string normalization wer is')
    x = 0
    for i in range(len(pr)):
        pr[i] = pr[i].lower().replace('.', '').replace(',', '').replace('-', '').replace('?', '').replace("'", '')
        gt[i] = gt[i].lower().replace('.', '').replace(',', '').replace('-', '').replace('?', '').replace("'", '')
        if pr[i] == gt[i]:
            x = x + 1
    post_wer = wer.compute(predictions=pr, references=gt)
    print('WER', post_wer)
    return_dict['post_ST_wer'] = post_wer
    print(x, '/', len(pr))
    return_dict['post_gtms'] = x / len(pr)
    print('*********************')
    return return_dict


adapter_path = '/path.to/your/adapter' # same as checkpoint_path

result_dict = result(adapter_path, model)
wer_percent = result_dict['WER'] * 100
wer_percent_post = result_dict['post_ST_wer'] * 100

gt_percent = result_dict['gtms'] * 100
gt_percent_post = result_dict['post_gtms'] * 100

print('epoch: ', checkpoint_path, 'WER: ', wer_percent, "WER_post: ", wer_percent_post, "GTM: ", gt_percent, "GTM_post: ", gt_percent_post)

