# VOiCES_utils
Scripts and utilities for working with the [VOiCES dataset](https://voices18.github.io/)

### Data

Instructions for downloading and descriptions of the VOiCES dataset can be found [here](https://voices18.github.io/).  The code in this repo is designed for the `VOiCES_devkit` and `VOiCES_release`.

## Contents

This repo is divided into several subdirectories, briefly described below.  Further details can be found in READMEs of the subdirectories.

### indexing_utils

This directory contains scripts for building up index files of the data set, or converting those index files into a format compatible with [Nvidia NeMo ASR](https://nvidia.github.io/NeMo/asr/tutorial.html#get-data).

### dataloaders

This directory contains class definitions for a PyTorch dataset that can be used to train a speaker verification model.

### ASR

This directory contains class definitions and scripts to facilitate ASR inference on VOiCES data using NeMo and [Quartznet](https://arxiv.org/abs/1910.10261).
