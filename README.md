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

### 3. Prepare the dataset for training (and inference)  
LipGER expects all train, val and test file to be in the format of [sample_data.json](./sample_data.json).
#### Step 1: Crop Mouth ROI from videos
- Install [RetinaFace](./crop_mouth) or [MediaPipe](https://pypi.org/project/mediapipe/) tracker.
- Run the following command from [crop_mouth](./crop_mouth) directory to crop mouth ROI:
  ```
  python crop_mouth.py data_filename=[data_filename] dst_filename=[dst_filename]
  ```
#### Step 2: Convert the cropped mouth ROI to hdf5 format
- LipGER expects a json with 'mouthroi' as one of the keys and the path to the mouth ROIs extracted in step 1. We have provided a sample json in [sample_data.json](./sample_data.json). Each instance in the json should look like below:
  
```
{
        "Dataset": "dataset_name",
        "Uid": "unique_id",
        "Caption": "[H] Okay. Yeah, I can deal with that.",
        "Noisy_Wav": "path_to_noisy_wav",
        "Mouthroi": "path_to_mouth_roi_mp4",
        "Video": "path_to_video_mp4",
        "nhyps_base": [ list of n-best hyps ],
        "whisper_emb": "whisper_emdedding_npy"
    }
```

- Update the path to json file (dataset_path) in [convert_lip.py](./crop_mouth/convert_lip.py), and run the following command:
```
python covert_lip.py
```
After converting the mp4 ROI to hdf5, the code will change the path of mp4 ROI to hdf5 ROI in the same json file. 

### 4. Fine-tune LipGER  

To finetune LipGER, just run:

```
sh finetune.sh
```
where you need to manually set the values for `data` (with the dataset name), `--train_path` and `--val_path` (with absolute paths to train and valid files).

## LipGER inference  

For inference just run:  

```
infer.sh
```

## 🌻 Acknowledgement  
The code for cropping mouth ROI is inspired from [Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages).

Our code for LipGER is inspired from [RobustGER](https://github.com/YUCHEN005/RobustGER/tree/master). Please cite their paper too if you find our paper or code useful.  


## 🔏 Citation    

```
@misc{ghosh2024lipger,
      title={LipGER: Visually-Conditioned Generative Error Correction for Robust Automatic Speech Recognition}, 
      author={Sreyan Ghosh and Sonal Kumar and Ashish Seth and Purva Chiniya and Utkarsh Tyagi and Ramani Duraiswami and Dinesh Manocha},
      year={2024},
      eprint={2406.04432},
      archivePrefix={arXiv},
      primaryClass={id='eess.AS' full_name='Audio and Speech Processing' is_active=True alt_name=None in_archive='eess' is_general=False description='Theory and methods for processing signals representing audio, speech, and language, and their applications. This includes analysis, synthesis, enhancement, transformation, classification and interpretation of such signals as well as the design, development, and evaluation of associated signal processing systems. Machine learning and pattern analysis applied to any of the above areas is also welcome.  Specific topics of interest include: auditory modeling and hearing aids; acoustic beamforming and source localization; classification of acoustic scenes; speaker separation; active noise control and echo cancellation; enhancement; de-reverberation; bioacoustics; music signals analysis, synthesis and modification; music information retrieval;  audio for multimedia and joint audio-video processing; spoken and written language modeling, segmentation, tagging, parsing, understanding, and translation; text mining; speech production, perception, and psychoacoustics; speech analysis, synthesis, and perceptual modeling and coding; robust speech recognition; speaker recognition and characterization; deep learning, online learning, and graphical models applied to speech, audio, and language signals; and implementation aspects ranging from system architecture to fast algorithms.'}
}
```
