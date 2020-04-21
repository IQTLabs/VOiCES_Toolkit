# Dataloaders

This directory contains helper classes necessary to instantiate a [PyTorch dataloader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
that can be used in a training pipeline for speaker identification.  This includes two components:

1. `VOiCES_SpeakerVerification`: A PyTorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) that can be used to load elements of the VOiCES dataset.  The subset of VOiCES (train, test, or some subset of either) referenced by this dataset is controlled by VOiCES index dataframe passed to the constructor.  This dataset returns a waveform and label for each element.  The label will either be sex (0,1) or speaker ID (an integer index into the set of unique speakers in the dataset).
2. `PadSequence`: A class which wraps a utility function for taking a batch of sequences of different lengths, padding them out to be the same length, stacking them into a tensor, and returning all of the information necessary to pass to [pack_padded_sequence](https://pytorch.org/docs/stable/nn.html#pack-padded-sequence) and create a [PackedSequence](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.PackedSequence) object.

## Example usage

```
from torch.nn.utils.rnn import pack_padded_sequence
from VOiCES_datasets import VOiCES_SpeakerVerification, PadSequence
import os

# load dataset dataframe
DATASET_ROOT = <path_to_dataset>
df = pd.read_csv(os.path.join(DATASET_ROOT,'references/test_index.csv'))

# instantiate dataset
voices = VOiCES_SpeakerVerification(DATASET_ROOT,df)

# instantiate the dataloader with batch size of 4
dataloader = DataLoader(voices,batch_size=4,shuffle=True,
  collate_fn=PadSequence())

for (wave,lengths,labels) in dataloader:
    print(wave.shape,lengths,labels)
```

This should produce output like the following

```
torch.Size([4, 267200, 1]) tensor([267200, 262720, 258160, 249760]) tensor([70, 88, 97, 91])
torch.Size([4, 273840, 1]) tensor([273840, 266480, 258320, 252560]) tensor([91, 19, 55,  0])
torch.Size([4, 264959, 1]) tensor([264959, 263040, 262800, 253920]) tensor([74, 10, 86, 45])
torch.Size([4, 260480, 1]) tensor([260480, 249040, 249040, 224000]) tensor([21, 69, 69, 99])
...
```
