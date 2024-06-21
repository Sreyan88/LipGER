import cv2
import numpy as np
import h5py
import json
import os
from tqdm import tqdm

def video_to_hdf5(video_path, hdf5_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Prepare the HDF5 file
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        # Check if the video has been opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            return

        # Get the total number of frames in the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get the height and width of frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a dataset in the HDF5 file
        dset = hdf5_file.create_dataset('video_frames', (frame_count, height, width), dtype='uint8')

        # Read each frame from the video
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Convert the frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Store the frame in the HDF5 file
                dset[frame_index] = gray_frame
                frame_index += 1
            else:
                break

    # Release the video capture object
    cap.release()

# Usage
dataset_path = '/fs/gamma-projects/audio/EasyComDataset_Processed/easycom_test_whisper.json'
dataset = open(dataset_path,'r')
dataset = json.load(dataset)
# all_dataset = []
for i in tqdm(dataset):
    video_path = i['Mouthroi']  # Replace with your video file path
    hdf5_path = video_path[:-4] + ".hdf5"  # Output HDF5 file
    i['Mouthroi'] = hdf5_path
    # all_dataset.append(i)
    if not os.path.exists(hdf5_path):
        print(hdf5_path)
        video_to_hdf5(video_path, hdf5_path)

# print(all_dataset[0])
with open(dataset_path, 'w') as dataset_file:
    json.dump(dataset, dataset_file, indent=4)
