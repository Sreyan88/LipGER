import whisper
from tqdm import tqdm
import os
import pdb
import pandas as pd
import collections
import json


df = pd.read_csv('/fs/gamma-projects/audio/microsoft/data/transcribed_data/en/asr_test.tsv',sep='\t')

model = whisper.load_model("tiny").to("cuda")
decode_options={"beam_size":10}


main_folder_path = "/fs/gamma-projects/audio/microsoft/data/transcribed_data/en/"
    
d= collections.defaultdict(list)

def save_json(data, filepath="microsoft_whisper_tiny.json"):
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)

processed_count=0
save_interval=100

# for root, dirs, files in os.walk(main_folder_path):
#     for filename in tqdm(files):
#         if filename.endswith(".ogg"):
#             try:
#                 path = os.path.join(root, filename)

#                 k= path
#                 result = model.transcribe(path, fp16=False,**decode_options)
#                 d[k] = []
#                 for i in range(10):
#                     d[k]+= [result[i]["text"]]

                
#                 processed_count += 1
#                 if processed_count%save_interval == 0:
                    
#                     #print("saved")
#                     save_json(d)
                    

#             except Exception as e:
#                 print(e)
#                 print("No")
#                 #print(filename)
# pdb.set_trace()
# with open("nhyps_pretrain_whisper.json", "w") as json_file:
#     json.dump(d, json_file, indent=4)


for filename in tqdm(list(df['id'])):
    folder = filename[:4]
    try:
        path = os.path.join(main_folder_path, folder + "/" + filename + '.ogg')
        k= path
        result = model.transcribe(path, fp16=False,**decode_options)
        d[k] = []
        for i in range(10):
            d[k]+= [result[i]["text"]]

        processed_count += 1
        if processed_count%save_interval == 0:            
            save_json(d)
            
    except Exception as e:
        print(e)

# pdb.set_trace()
with open("mciro_whisper_tiny.json", "w") as json_file:
    json.dump(d, json_file, indent=4)

