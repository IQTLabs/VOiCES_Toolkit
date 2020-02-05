# VOiCES_utils
Scripts and utilities for working with the VOiCES dataset

## Contents
## Instructions

### Training and Evaluating End-to-End ASR model on VOiCES with Nvidia NeMo

```
# Pull the docker
docker pull nvcr.io/nvidia/pytorch:19.11-py3


# Run Docker for docker version >=19.03, and mount the folder containing
# the VOiCES dataset
docker run --gpus all -it -v <nemo_github_folder>:/NeMo -v <dataset_folder>:/data  --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.11-py3

#Change to the nemo folder
cd /NeMo

pip install nemo-toolkit  # installs NeMo Core
pip install nemo-asr # installs NeMo ASR collection
pip install .
pip install frozendict
```

## Requirements
