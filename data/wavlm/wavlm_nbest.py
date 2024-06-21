from espnet2.bin.asr_inference import Speech2Text
import soundfile
import json
from tqdm import tqdm
import argparse
import torchaudio

model = Speech2Text.from_pretrained(
  "espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp",
  device='cuda',
  predict_time=True
)
model.nbest = 10

parser = argparse.ArgumentParser(description='Transcribe audio files using WavLM.')
parser.add_argument('--file_path', type=str, required=True, help='Path to the file containing audio file paths')

# Parse arguments
args = parser.parse_args()

files = open(args.file_path,'r')
files = json.load(files)
files = [i['Clean_Wav'] for i in files]
files = [i.strip() for i in files]
files = [i.strip() for i in files if i.endswith('.wav')]
print("Total files to process:",len(files))
print(model.device)


d = dict()

for file in tqdm(files):

  try:
    speech, rate = soundfile.read(file)
    output = model(speech)
    texts = []
    for i in range(10):
        texts.append(output[i][0])
    if len(texts) == 0:
      print(file)
    d[file] = texts
  except Exception as e:
    print(e)

with open('dataset.json','w') as jsonfile:
   json.dump(d, jsonfile, indent=4)


