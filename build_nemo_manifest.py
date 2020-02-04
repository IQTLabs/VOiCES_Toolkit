import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',dest='DATASET_ROOT',help='VOiCES dataset root',
    default='none',type=str)
    parser.add_argument('-i',dest='INDEX_PATH',help='The absolute path to the index file',
    default='none',type=str)
    parser.add_argument('-o',dest='MANIFEST_PATH',help='Target file for the built manifest',
    default='none',type=str)
    parser.add_argument('--drop_bad',dest='DROP_BAD',action='store_true')
    args = parser.parse_args()

    DATASET_ROOT=args.DATASET_ROOT
    if args.DATASET_ROOT == 'none':
        DATASET_ROOT = os.path.join(os.getcwd(),'')
    else:
        DATASET_ROOT = os.path.join(args.DATASET_ROOT,'')

    if args.INDEX_PATH=='none':
        INDEX_PATH = os.path.join(DATASET_ROOT, 'references/train_index.csv')
    else:
        INDEX_PATH = args.INDEX_PATH

    # load the index file
    df = pd.read_csv(INDEX_PATH,index_col='index')
    # If DROP_BAD == True, drop all recordings of incorrect length
    if args.DROP_BAD:
        df =df[df['noisy_length']==df['source_length']]

    # Keep only the filename, noisy recording time, and transcript columns
    df = df[['filename','noisy_time','transcript']]
    # Rename columns
    df.rename(columns={"filename":"audio_filepath",
    "noisy_time":"duration","transcript":"text"})
    # Add the dataset root to every every entry in the filename column
    df['filename'] = df['filename'].apply(lambda x: DATASET_ROOT+x)
    # Output to JSON
    df.to_json(args.MANIFEST_PATH,orient='records',lines=True)
