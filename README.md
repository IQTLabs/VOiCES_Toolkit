# VOiCES_utils
Scripts and utilities for working with the VOiCES dataset

## Contents
## Instructions

### Training and Evaluating End-to-End ASR model on VOiCES with Nvidia NeMo

```
# Pull the docker
docker pull nvcr.io/nvidia/pytorch:19.11-py3

# Run Docker for docker version >=19.03, and mount folders containing the NeMo
# repo, the datasets (with the root folders as subfolders), and the VOiCES_utils
# repo.
docker run --gpus all -it -v <nemo_github_folder>:/NeMo -v <dataset_folder>:/data -v <VOiCES_utils_folder>:/utils --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.11-py3

#Change to the nemo folder
cd /NeMo

pip install nemo-toolkit  # installs NeMo Core
pip install nemo-asr # installs NeMo ASR collection
pip install .
pip install frozendict
```

```
python /NeMo/examples/asr/jasper_eval.py --model_config=/NeMo/examples/asr/configs/quartznet15x5.yaml --eval_datasets "/data/VOiCES_devkit/references/test_manifest_mic_5_dist_none.json" --load_dir=/data/quartznet_checkpoints --lm_path=/NeMo/scripts/language_model/6-gram-lm.binary --batch_size=16
```

## Requirements
