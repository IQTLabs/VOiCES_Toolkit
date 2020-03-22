import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import librosa

class VOiCES_SpeakerVerfication(Dataset):
    """
    A torch dataset class for speaker/gender classification tasks.
    
    __getitem__ returns the raw waveform (by default) and the integer label
    
    # Arguments:
        dataset_root: The path to the root of the VOiCES dataset
        df: A dataframe indexing the VOiCES dataset, may be a subset of the full dataset
        min_length: mininum length, in seconds, of recordings to include
        max_length: mininum length, in seconds, of recordings to include
        label:  One of {speaker, sex}. Whether to use sex or speaker ID as a label
        transform:  Callable, transformation to perform on the waveform
    """
    def __init__(self,dataset_root,df,min_length=0.0,max_length=30.0,label='speaker',transform=None):
        if label not in ('sex','speaker'):
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')')
        self.default_samplerate=16000
        self.label = label
        
        self.dataset_root = dataset_root
        
        # trim out all recordings that are too short or too long
        df = df[df['noisy_time']>=min_length]
        df = df[df['noisy_time']<=max_length]
        self.df = df
        
        # sort out unique speakers
        self.unique_speakers = sorted(self.df['speaker'].unique())
        self.num_speakers = len(self.unique_speakers)
        self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_speakers)}
        
        
        self.transform = transform
        
    def __getitem__(self,index):
        item = self.df.iloc[index]
        filepath = os.path.join(self.dataset_root,item['filename'])
        instance,samplerate = librosa.load(filepath,sr=self.default_samplerate)
        
        if self.label == 'sex':
            gender = item['gender']
            if gender == 'M':
                label = 0
            elif gender == 'F':
                label = 1
        elif self.label =='speaker':
            label = self.speaker_id_mapping[item['speaker']]
            
        # Add transforms
        if self.transform is not None:
            instance = self.transform(instance)
        return instance,label
    
    def __len__(self):
        return len(self.df)

    def num_classes(self):
        if self.label=='speaker':
            return len(self.df['speaker_id'].unique())
        else:
            return 2
