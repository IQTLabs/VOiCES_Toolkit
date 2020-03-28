# Indexing scripts

This directory contains two scripts that are useful creating indices of the
dataset `build_indices.py` and `build_nemo_manifest.py`

## build_indices.py

This script parses the directory tree of the VOiCES dataset and produces two
csv files,`train_index.csv` for the training set and `test_index.csv` for the
test set, with one row for each recording and the following columns:

|Column   |Datatype   |Description|   
|---------|-----------|-----------|
|index| integer| Unique index for recording
|chapter| integer | Librispeech chapter ID
|degrees| integer | Angle (in degrees) between source speaker and mic
|distractor| string | Distractor type, options are 'none', 'babb', 'tele, 'musi'
|filename| string| Path to recording .wav, relative to root directory
|gender| string| Speaker gender, options are 'M' and 'F'
|mic| integer| The mic used for this recording
|query_name| string| The filename without directory path or extension
|room| string| The room recorded in, options are 'rm1', 'rm2', 'rm3', 'rm4'
|segment| integer| Librispeech segment ID
|source| string| Path to .wav file for Librispeech source audio for this recording
|speaker| integer| Librispeech speaker ID
|transcript| string| Orthographic transcript of the Librispeech source audio
|noisy_length| integer| Sample length of recording
|noisy_sr| integer| Sample rate (hz) of recording
|noisy_time| float| Duration of recording in seconds
|source_length| integer| Sample length of Librispeech source audio
|source_sr| integer| Sample rate (hz) of Librispeech source audio
|source_time| float| Duration of Librispeech source audio in seconds

To build these index files run the following command

```
python build_indices.py -r <path_to_voices_root> -i <path_to_index_location>
```

* `<path_to_voices_root>` is the absolute path to the root of the voices
file directory  
* `<path_to_index_location>` should be the absolute path to
the directory where the index files should be location, which defaults .

## build_nemo_manifest.py

This script takes in an index file and produces a json file in a format compatible
with [Nvidia NeMo ASR](https://nvidia.github.io/NeMo/asr/tutorial.html#get-data).

```
python build_nemo_manifest.py -r <path_to_voices_root> -i <path_to_csv>
-o <path_to_json_output> -m <max_duration> --drop_bad --split
```
* `<path_to_voices_root>` is the absolute path to the root of the voices
file directory
* `<path_to_csv>` is the path to index csv file
* `<path_to_json_output>` is the absolute path to the output json file, must include the `.json` extension
* `<max_duration>` is the maximum length (in seconds) of recordings to include in the dataset.
* `--drop_bad` is an optional argument that will drop VOiCES recordings which do not match the length of the original Librispeech source audio.
* `--split` will divide the dataset by distractor type and mic number and produce a separate json file for each combination of mic and distractor.
