"""
This script will produce two csv files that can serves as index files for the
training and test splits of the VOiCES data.

The script takes in two command line arguments.

-r: The absolute path of the dataset root (where there are subfolders /references,
/distant-16k, and /source-16k).  Defaults to current working directory
-i: The path to place the output csv files.  Defaults to /references subfolder
of the dataset root.

There are two output csv files, both with the same column structure (explained
below):

train_index.csv : A csv file with a row for every entry in the train split
test_index.csv : A csv file with a row for every entry in the test splits

Both outputs have the following set of columns:

index: The entry index (int)
chapter: The librispeech chapter ID for the transcript (int)
degrees: The angle, in degrees
between output speaker and microphone for this entry (int)
distractor: The type of background distractor noise.  One of ['musi','tele',
'babb', 'none'] (string)
filename: The filepath for the recording, relative to the dataset root (string)
gender: The gender of the speaker.  One of ['M','F'] (string)
mic:  The mic ID.  For rooms 1 and 2 this is in [1,...,12] and for rooms 3 and 4
this is in [1,...,20] (int)
query_name: The filename with the path info removed (string)
room: Which room the recording is from.  One of ['rm1','rm2','rm3','rm4'] (string)
segment:  The librispeech segment ID for the transcript (int)
source:  The filepath for the source audio, relative to the dataset root (string)
speaker:  The librispeech speaker ID (int)
transcript:  The orthographic transcript of the source audio (string)
noisy_length:  The length of the recording in samples (int)
noisy_sr:  The sampling rate of the recording in Hz, should be 16000 (int)
noisy_time:  The length of the recording in seconds (float)
source_length: The length of the source audio in samples (int)
source_sr: The sampling rate of the source audio in Hz, should be 16000 (int)
source_time: The length of the source audio in seconds (float)

"""

import os
import argparse
import pandas as pd

def parse_file(filename):
    """
    Returns a dictionary containing parsed information from the .wav filename
    """
    # Strip off path
    wav_name = filename.split('/')[-1]

    # file_info will be a dictionary containing information from the filename
    file_info = {'filename':filename}
    file_info['query_name']=file_info['filename'].split('/')[-1].split('.')[0]
    # Get room information
    rm_ind = wav_name.find('rm')
    file_info['room'] = wav_name[rm_ind:rm_ind+3]
    # Get distractor information
    for dist in ['babb','musi','none','tele']:
        if dist in wav_name:
            file_info['distractor'] = dist
            break
    # Get speaker
    sp_ind = wav_name.find('sp')
    file_info['speaker'] = int(wav_name[sp_ind+2:sp_ind+6])

    # Get chapter
    ch_ind =wav_name.find('ch')
    file_info['chapter'] = int(wav_name[ch_ind+2:ch_ind+8])

    # Get segment
    sg_ind = wav_name.find('sg')
    file_info['segment'] = int(wav_name[sg_ind+2:sg_ind+6])

    #Get mic
    mc_ind = wav_name.find('mc')
    file_info['mic'] = int(wav_name[mc_ind+2:mc_ind+4])

    # Get degrees
    dg_ind = wav_name.find('dg')
    file_info['degrees'] = int(wav_name[dg_ind+2:dg_ind+5])
    return file_info
def add_gender(file_info,speaker_gender_df):
    """
    Adds a gender entry to the dictionary of file information
    """
    gender = speaker_gender_df[speaker_gender_df['Speaker']==file_info['speaker']]['Gender'].values[0]
    new_info = dict(file_info)
    new_info['gender']=gender
    return new_info
def get_source_file(noisy_spch):
    """
    Retrieves the original source file
    """
    if 'train' in noisy_spch:
        train_test = 'train'
    elif 'test' in noisy_spch:
        train_test = 'test'
    else:
        raise ValueError('File was not in train or test directory')
    speaker = noisy_spch[noisy_spch.find('-sp')+1:noisy_spch.find('-ch')]
    src_file = 'source-16k/'+train_test+'/'+speaker+'/'+'Lab41-SRI-VOiCES-src'+noisy_spch[noisy_spch.find('-sp'):noisy_spch.find('-mc')]+'.wav'
    return src_file
def full_pipeline(filename,speaker_gender_df):
    file_info = parse_file(filename)
    file_info = add_gender(file_info,speaker_gender_df)
    return file_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',dest='DATASET_ROOT',help='VOiCES dataset root',
                        default='none',type=str)
    parser.add_argument('-i',dest='INDEX_PATH',help='Target directory for index files',
                        default='none',type=str)
    args = parser.parse_args()

    if args.DATASET_ROOT == 'none':
        DATASET_ROOT = os.path.join(os.getcwd(),'')
    else:
        DATASET_ROOT = os.path.join(args.DATASET_ROOT,'')

    if args.INDEX_PATH=='none':
        INDEX_PATH = os.path.join(DATASET_ROOT, 'references/')
    else:
        INDEX_PATH = os.path.join(args.INDEX_PATH,'')

    name_start = len(DATASET_ROOT)
    # Gather reference files
    # All of these files should be in the references subfolder of the dataset
    speaker_gender_df = pd.read_table(DATASET_ROOT + 'references/' + 'Lab41-SRI-VOiCES-speaker-gender-dataset.tbl', sep='\s+')
    speaker_chapter_df = pd.read_table(DATASET_ROOT + 'references/' + 'Lab41-SRI-VOiCES-speaker-book-chapter.tbl', sep='\s+')
    full_ref_df = pd.read_csv(DATASET_ROOT + 'references/' + 'filename_transcripts',index_col='index')
    time_df = pd.read_csv(DATASET_ROOT + 'references/' + 'time_values.csv',index_col='index')
    time_df2=time_df[['noisy_filename','noisy_length','noisy_sr','noisy_time'
                  ,'source_length','source_sr','source_time']]
    time_df2=time_df2.rename(columns={"noisy_filename":"filename"})

    # Find all files in training set
    print('Scraping Training Files')
    train_file_list = []
    for root, dirs, files in os.walk(DATASET_ROOT+'distant-16k/speech/train'):
        for name in files:
            train_file_list.append(os.path.join(root, name)[name_start:])
    # Parse all files in training set
    print('Building index for training set, this may take several minutes')
    train_info = []
    for name in train_file_list:
        info_dict = full_pipeline(name,speaker_gender_df)
        source_filename = get_source_file(name)
        info_dict['source']=source_filename
        train_info.append(info_dict)
    train_index = pd.DataFrame(train_info)
    # Add transcripts to index
    train_index = train_index.join(full_ref_df.set_index('file_name'),on='query_name')
    # Add precomputed information on the lengths of the files
    train_index = train_index.join(time_df2.set_index('filename'),on='filename')
    # Save
    train_index.to_csv(path_or_buf = INDEX_PATH+'train_index.csv',index_label='index')

    #Find all files in test set
    print('Scraping Testing Files')
    test_file_list = []
    for root, dirs, files in os.walk(DATASET_ROOT+'distant-16k/speech/test'):
        for name in files:
            test_file_list.append(os.path.join(root, name)[name_start:])
    # Parse all files in test set
    print('Building index for test set, this make take several minutes')
    test_info = []
    for name in test_file_list:
        info_dict = full_pipeline(name,speaker_gender_df)
        source_filename = get_source_file(name)
        info_dict['source']=source_filename
        test_info.append(info_dict)
    test_index = pd.DataFrame(test_info)
    # Add transcripts to index
    test_index = test_index.join(full_ref_df.set_index('file_name'),on='query_name')
    # Add precomputed information on the lengths of the files
    test_index = test_index.join(time_df2.set_index('filename'),on='filename')
    # Save
    test_index.to_csv(path_or_buf = INDEX_PATH+'test_index.csv',index_label='index')
