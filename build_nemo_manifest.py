import os
import argparse
import pandas as pd

def trim_df(df,max_duration=30.0,drop_bad=True):
    """
    Performs some cleaning on the dataframe, dropping recordings that
    are too long, or where the source and noisy recording lengths differ

    Inputs:
        df - A pandas dataframe representing the index file of the dataset,
            with the default columns of VOiCES index files.
        max_duration - The maximum duration, in seconds, of recordings to keep
            in the dataset (float)
        drop_bad - If True, entries where the noisy recording is not the same
            length as the source audio will be dropped
    Outputs:
        new_df - The cleaned dataframe
    """
    if drop_bad:
        new_df = df[df['noisy_length']==df['source_length']]
    else:
        new_df = df.copy()
    new_df = new_df[new_df['noisy_time']<=max_duration]
    return new_df

def convert_df_to_manifest(df,dataset_root):
    """
    Converts a dataframe in the format of a VOiCES index file to a JSON string
    compatible with the NeMo ASR manifest format.

    Inputs:
    df - A pandas dataframe representing the index file of the dataset,
        with the default columns of VOiCES index files.
    dataset_root - A string with the absolute path to the root folder of the
        VOiCES dataset.
    Outputs:
    record_string - A string representing the nemo manifest, in newline
        delimited format.
    """
    # Keep only the filename, noisy recording time, and transcript columns
    df = df[['filename','noisy_time','transcript']]
    # Rename columns
    df=df.rename(columns={"filename":"audio_filepath",
    "noisy_time":"duration","transcript":"text"})
    # Add the dataset root to every every entry in the filename column
    df['audio_filepath'] = df['audio_filepath'].apply(lambda x: os.path.join(dataset_root,x))
    # Output to JSON
    record_string = df.to_json(orient='records',lines=True)
    return record_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',dest='DATASET_ROOT',help='VOiCES dataset root',
    default='none',type=str)
    parser.add_argument('-i',dest='INDEX_PATH',help='The absolute path to the index file',
    default='none',type=str)
    parser.add_argument('-o',dest='MANIFEST_PATH',help='Target file for the built manifest',
    default='none',type=str)
    parser.add_argument('-m',dest='MAX_DURATION',help='Maximum allowed recording duration in seconds',
    default=30.0,type=float)
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

    trimmed_df = trim_df(df,max_duration=args.MAX_DURATION,
    drop_bad=args.DROP_BAD)
    json_string = convert_df_to_manifest(trimmed_df,DATASET_ROOT)
    # Output to file
    with open(args.MANIFEST_PATH,'w') as fout:
        fout.write(json_string)
