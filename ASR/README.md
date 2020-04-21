# ASR
This directory contains utilities for performing inference with [Quartznet](https://arxiv.org/abs/1910.10261) on VOiCES recordings
using the [NVIDIA Neural Modules toolkit](https://nvidia.github.io/NeMo/index.html).  The contents are summarized below, with more details on inputs and outputs found in the files.

### `infer_datalayers.py`

This file contains the class definition for `AudioInferDataLayer`, a [NeMo Datalayer](http://nemo-master-docs.s3-website.us-east-2.amazonaws.com/api-docs/nemo.html#nemo.backends.pytorch.nm.DataLayerNM) that supports easy programmatic interaction for loading specific waveforms into an ASR pipeline.

### `JasperModels.py`

This file contains the class definition for `JasperInference`.  This class wraps `AudioInferDataLayer` and several other Neural Modules, and provides an `infer` method to perform inference on a user supplied list of waveforms or `.wav` filepaths.

### `batch_asr_eval.py`

This script uses `JasperInference` and takes in a VOiCES index csv file and performs inference on all the recordings indexed by that file.

Example use:

```
python batch_asr_eval -r <path_to_dataset_root> -i
<path_to_index.csv> -e <path_to_decoder_weights.pt> -d
<path_to_decoder_weights.pt> -c <path_to_jasper_config.yml>
-o <path_to_output_file.csv> -b <batch_size>
```
