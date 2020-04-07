import os
import argparse
import pandas as pd
import pesq
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed

def process_item(item,dataset_root,sample_rate=16000):
    result_dict = {'query_name':item['query_name']}

    noisy_filepath = os.path.join(dataset_root,item['filename'])
    clean_filepath = os.path.join(dataset_root,item['source'])

    noisy_waveform,_ = librosa.load(noisy_filepath,sr=sample_rate)
    clean_waveform,_ = librosa.load(clean_filepath,sr=sample_rate)

    if len(noisy_waveform)==len(clean_waveform):
        pesq_nb = pesq.pesq(16000,clean_waveform,noisy_waveform,'nb')
        pesq_wb = pesq.pesq(16000,clean_waveform,noisy_waveform,'wb')
        result_dict['pesq nb'] = pesq_nb
        result_dict['pesq wb'] = pesq_wb
    else:
        result_dict['pesq nb'] = -1
        result_dict['pesq wb'] = -1

    return result_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',dest='DATASET_ROOT',help='VOiCES dataset root',
                        default='none',type=str)
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.DATASET_ROOT,'references/train_index.csv'))
    test_df = pd.read_csv(os.path.join(args.DATASET_ROOT,'references/test_index.csv'))
    test_df = test_df.reset_index(drop=True)
    full_df = pd.concat([train_df,test_df],axis=0)

    #convert the dataframe to a list of dicts
    records = full_df.to_dict('records')

    process_func = lambda record: process_item(record,args.DATASET_ROOT)
    result_list = Parallel(n_jobs=8)(delayed(process_func)(item) for item in tqdm(records))
    result_df = pd.DataFrame(result_list)
    result_df.to_csv('full_pesq.csv')
