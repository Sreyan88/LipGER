## Installing Retinaface
We use [ibug.face_detection](https://github.com/hhj1897/face_detection) and [ibug.face_alignment](https://github.com/hhj1897/face_alignment) in this repository.

### Prerequisites
* [Git LFS](https://git-lfs.github.com/), needed for downloading the pretrained weights that are larger than 100 MB.

### From source

1. Install *`ibug.face_detection`*
```Shell
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
```

2. Install *`ibug.face_alignment`*
```
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```